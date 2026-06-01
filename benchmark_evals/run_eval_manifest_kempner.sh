#!/bin/bash
#
# Slurm array driver for manifest-based benchmark evals.
# Set NUM_WORKERS (matches --array=0-$((N-1))) and MANIFEST before sbatch'ing.
# Each worker takes its slice of the manifest via worker_id % num_workers.
#
# Usage:
#   NUM_WORKERS=8 MANIFEST=/path/to/manifest.json \
#     sbatch --array=0-7 run_eval_manifest_kempner.sh

#SBATCH --job-name=bench-eval-manifest
#SBATCH --account=kempner_barak_lab
#SBATCH --partition=kempner
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH -o /n/home05/sqin/OLMo-core/slurm_out/bench-eval-manifest-%A_%a.out
#SBATCH -e /n/home05/sqin/OLMo-core/slurm_out/bench-eval-manifest-%A_%a.out

set -euo pipefail

: "${MANIFEST:?MANIFEST env var must point to the manifest.json}"
: "${NUM_WORKERS:?NUM_WORKERS env var must match --array size}"

module purge
module load Mambaforge
module load cuda cudnn

PYTHON=/n/holylabs/dam_lab/Lab/sqin/envs/openrlhf/bin/python

echo "Worker $SLURM_ARRAY_TASK_ID / $NUM_WORKERS  manifest=$MANIFEST"

HF_DIR=/tmp/hf_models_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$HF_DIR"
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$TRITON_CACHE_DIR"

cd /n/home05/sqin/OLMo-core

"$PYTHON" -u benchmark_evals/convert_and_eval.py \
  --manifest "$MANIFEST" \
  --worker-id "$SLURM_ARRAY_TASK_ID" \
  --num-workers "$NUM_WORKERS" \
  --limit 1000 \
  --output-dir "$HF_DIR" \
  --results-dir /n/home05/sqin/OLMo-core/benchmark_evals/results

echo "Worker $SLURM_ARRAY_TASK_ID done."
