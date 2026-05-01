#!/bin/bash
# -----------------------------------------------------------------------------
# Build size-aligned paraphrase shards for the 14M / 30M / 60M / 190M ladders.
#
# Job array: one task per V2 seed (1..8). Each task builds all 24 unique target
# shards (the union of model ladders) for its seed. The Python builder is
# resume-safe: existing output files are skipped unless --overwrite is set.
#
# Submit:
#   sbatch experiment_scripts/paraphrasing/submit_build_sized_paraphrase.sh
# -----------------------------------------------------------------------------

#SBATCH -p seas_compute
#SBATCH -t 4:00:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -n 1
#SBATCH --array=1-8
#SBATCH -o /n/home05/sqin/OLMo-core/slurm_out/build_sized_paraphrase-%A_%a.out
#SBATCH -e /n/home05/sqin/OLMo-core/slurm_out/build_sized_paraphrase-%A_%a.out
#SBATCH --job-name=para-resize

set -euo pipefail

ENV_PYTHON="/n/holylabs/dam_lab/Lab/sqin/envs/openrlhf/bin/python"
cd /n/home05/sqin/OLMo-core

SEED="${SLURM_ARRAY_TASK_ID}"

# Union of unique shard names from the 14M, 30M, 60M, 190M ladders in
# src/scripts/official/OLMo-scale-train-paraphrase-dolma.py (line 81+).
# 24 unique shards.
SHARDS="train_0.014B,train_0.028B,train_0.03B,train_0.06B,train_0.07B,train_0.12B,train_0.14B,train_0.15B,train_0.19B,train_0.28B,train_0.3B,train_0.38B,train_0.56B,train_0.6B,train_0.95B,train_1.12B,train_1.2B,train_1.9B,train_2.24B,train_2.4B,train_3.8B,train_4.48B,train_4.8B,train_7.4B"

echo "=========================================="
echo "Build sized paraphrase shards"
echo "  Seed       : $SEED"
echo "  Shards (24): $SHARDS"
echo "  Partition  : $SLURM_JOB_PARTITION"
echo "  Node       : $(hostname)"
echo "  Job        : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  Started at : $(date)"
echo "=========================================="

"$ENV_PYTHON" -u experiment_scripts/paraphrasing/build_sized_paraphrase.py \
    --shards "$SHARDS" \
    --seeds "$SEED" \
    --overwrite

echo ""
echo "Finished at $(date)"
