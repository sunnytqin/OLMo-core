#!/bin/bash
# Merge individual eval results and generate heatmaps.
# Run this after all eval jobs complete.

set -e

EVALS_DIR="../dolma_val_loss"
MERGED_DIR="merged"
HEATMAPS_DIR="heatmaps"

MODEL_SIZES=("14M" "30M" "60M" "190M" "370M")
CHIN_DIRS=(chinchilla_0.05 chinchilla_0.1 chinchilla_0.25 chinchilla_0.5 chinchilla_1 chinchilla_2 chinchilla_4 chinchilla_8 chinchilla_16)

mkdir -p "$MERGED_DIR"

echo "=== Merging individual eval results ==="
for chin in "${CHIN_DIRS[@]}"; do
    for size in "${MODEL_SIZES[@]}"; do
        indir="$EVALS_DIR/$chin/$size"
        if [ -d "$indir" ]; then
            outfile="$MERGED_DIR/${chin}_${size}.json"
            echo "  Merging $chin/$size -> $outfile"
            python merge_eval_results.py "$indir" "$outfile"
            echo ""
        fi
    done
done

echo ""
echo "=== Generating heatmap grids ==="
python chinchilla_heatmaps.py --merged-dir "$MERGED_DIR" --output-dir "$HEATMAPS_DIR"

echo ""
echo "=== Generating optimal hparam plots ==="
python optimal_hparams_plot.py --merged-dir "$MERGED_DIR" --output-dir "$HEATMAPS_DIR"

echo ""
echo "=== Done ==="
echo "Merged results: $MERGED_DIR"
echo "Heatmaps: $HEATMAPS_DIR"
