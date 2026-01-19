#!/bin/bash

# Configuration
BASE_URL="http://olmo-data.org"
OUTPUT_DIR="/n/netscratch/dam_lab/Lab/sqin/olmo"

# File lists with their corresponding tokenizer paths
# Training data uses "allenai/dolma2-tokenizer" (with directory structure)
# Eval data uses "dolma2-tokenizer" (flat naming)
declare -A FILE_LISTS
FILE_LISTS["src/olmo_core/data/mixes/OLMo-dclm-sample.txt"]="allenai/dolma2-tokenizer"
FILE_LISTS["src/olmo_core/data/mixes/v3-small-ppl-validation.txt"]="dolma2-tokenizer"

NUM_PARALLEL_JOBS=8  # Adjust based on your network and system capacity
LOG_FILE="download_progress.log"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to download a single file
download_file() {
    local url="$1"
    local output_path="$2"

    # Create directory for the file if it doesn't exist
    mkdir -p "$(dirname "$output_path")"

    # Download with resume capability (-c), quiet mode except for errors
    # Use --no-clobber to skip already completed downloads
    if [ -f "$output_path" ]; then
        # File exists, check if it's complete by trying to download and comparing
        wget -c -q --show-progress "$url" -O "$output_path" 2>&1
    else
        wget -c -q --show-progress "$url" -O "$output_path" 2>&1
    fi

    local status=$?
    if [ $status -eq 0 ]; then
        echo "✓ $(basename "$output_path")"
        return 0
    else
        echo "✗ $(basename "$output_path") (exit code: $status)"
        return 1
    fi
}

export -f download_file
export BASE_URL OUTPUT_DIR

# Process the file lists and create download jobs
echo "Reading file lists:"
for file_list in "${!FILE_LISTS[@]}"; do
    echo "  - $file_list (tokenizer: ${FILE_LISTS[$file_list]})"
done

# Count total files across all lists
total_files=0
for file_list in "${!FILE_LISTS[@]}"; do
    count=$(grep -v '^#' "$file_list" | grep -v '^[[:space:]]*$' | wc -l)
    total_files=$((total_files + count))
done

echo "Found $total_files files to download"
echo "Saving to: $OUTPUT_DIR"
echo "Using $NUM_PARALLEL_JOBS parallel jobs"
echo ""

# Create a temporary file with all URLs and output paths
temp_file=$(mktemp)
trap "rm -f $temp_file" EXIT

# Process all file lists
for file_list in "${!FILE_LISTS[@]}"; do
    tokenizer="${FILE_LISTS[$file_list]}"
    grep -v '^#' "$file_list" | grep -v '^[[:space:]]*$' | while IFS=',' read -r prefix path; do
        # Replace {TOKENIZER} placeholder with the appropriate tokenizer for this file list
        path_expanded="${path//\{TOKENIZER\}/$tokenizer}"

        # Construct full URL
        url="${BASE_URL}/${path_expanded}"

        # Construct output path (preserve directory structure)
        output_path="${OUTPUT_DIR}/${path_expanded}"

        # Output as tab-separated values
        echo -e "${url}\t${output_path}"
    done
done > "$temp_file"

echo "Starting parallel downloads..."
echo "Progress will be shown below. Detailed log: $LOG_FILE"
echo ""

# Use GNU parallel to download files
# --colsep '\t': tab-separated input
# -j: number of parallel jobs
# --bar: show progress bar
# --joblog: log file for tracking progress
# --resume: resume from joblog if script is restarted
cat "$temp_file" | parallel \
    --colsep '\t' \
    -j "$NUM_PARALLEL_JOBS" \
    --bar \
    --joblog "$LOG_FILE" \
    --resume \
    --retries 3 \
    download_file {1} {2}

echo ""
echo "Download complete!"
echo ""

# Show summary
success_count=$(grep -c "^✓" "$LOG_FILE" 2>/dev/null || echo "0")
failed_count=$(awk '$7 != "0" && NR > 1' "$LOG_FILE" 2>/dev/null | wc -l)

echo "Summary:"
echo "  Total files: $total_files"
echo "  Successful: $success_count"
echo "  Failed: $failed_count"
echo ""
echo "Check $LOG_FILE for detailed job log"

if [ "$failed_count" -gt 0 ]; then
    echo ""
    echo "To retry failed downloads, simply run this script again."
    echo "It will automatically resume from where it left off."
fi
