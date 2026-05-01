#!/bin/bash
# Merge individual eval results and generate heatmaps for one or more settings.
#
# Usage:
#   bash merge_and_plot.sh                  # both settings
#   bash merge_and_plot.sh multi_epoch      # just multi-epoch
#   bash merge_and_plot.sh para             # just paraphrase
#   bash merge_and_plot.sh multi_epoch para # both, explicit

set -e

MODEL_SIZES=("14M" "30M" "60M" "190M" "370M")
CHIN_DIRS=(chinchilla_0.05 chinchilla_0.1 chinchilla_0.25 chinchilla_0.5 chinchilla_1 chinchilla_2 chinchilla_4 chinchilla_8 chinchilla_16)

# Per-setting source directory (raw eval JSONs).
declare -A EVAL_SRC_DIR
EVAL_SRC_DIR[multi_epoch]="../dolma_val_loss"
EVAL_SRC_DIR[para]="../dolma_para_val_loss"

run_setting() {
    local setting="$1"
    local src="${EVAL_SRC_DIR[$setting]}"
    local merged_dir="merged/${setting}"
    local heatmaps_dir="heatmaps/${setting}"

    if [ -z "$src" ]; then
        echo "Unknown setting: $setting" >&2
        exit 1
    fi

    mkdir -p "$merged_dir" "$heatmaps_dir"

    echo ""
    echo "############################################################"
    echo "# Setting: $setting"
    echo "# Source:  $src"
    echo "# Merged:  $merged_dir"
    echo "# Heatmaps:$heatmaps_dir"
    echo "############################################################"

    echo "=== Merging individual eval results ($setting) ==="
    for chin in "${CHIN_DIRS[@]}"; do
        for size in "${MODEL_SIZES[@]}"; do
            indir="$src/$chin/$size"
            if [ -d "$indir" ]; then
                outfile="$merged_dir/${chin}_${size}.json"
                echo "  Merging $chin/$size -> $outfile"
                python merge_eval_results.py "$indir" "$outfile"
                echo ""
            fi
        done
    done

    echo "=== Generating heatmap grids ($setting) ==="
    python chinchilla_heatmaps.py --setting "$setting" \
        --merged-dir "$merged_dir" --output-dir "$heatmaps_dir"

    echo ""
    echo "=== Generating optimal hparam plots ($setting) ==="
    python optimal_hparams_plot.py --setting "$setting" \
        --merged-dir "$merged_dir" --output-dir "$heatmaps_dir"
}

if [ "$#" -eq 0 ]; then
    SETTINGS_TO_RUN=(multi_epoch para)
else
    SETTINGS_TO_RUN=("$@")
fi

for setting in "${SETTINGS_TO_RUN[@]}"; do
    run_setting "$setting"
done

echo ""
echo "=== Done ==="
