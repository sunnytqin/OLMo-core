#!/bin/bash
#
# Submit a self-distillation generation job. Two parallelism modes:
#
#   1. Per-teacher (NUM_SHARDS=1, default): one job generates all N_target files
#      sequentially. Resume after timeout via per-file checkpoint.
#
#   2. Sharded job array (NUM_SHARDS=N + sbatch --array=0-{N-1}): each array
#      task generates total_tokens/N tokens, gets its SHARD_ID from
#      SLURM_ARRAY_TASK_ID, and writes to the same flat output dir with
#      shard_id-prefixed file names so shards never collide.
#
# Usage:
#     MODEL_SIZE=30M CHIN=0.5 sbatch submit_generation.sh
#     MODEL_SIZE=30M CHIN=0.5 NUM_SHARDS=16 sbatch --array=0-15%16 submit_generation.sh
#     MODEL_SIZE=30M CHIN=0.5 NUM_SHARDS=16 SHARD_ID=5 sbatch submit_generation.sh  # single shard re-run

#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00
#SBATCH --mem=64G
#SBATCH --job-name=syn-gen
#SBATCH -o /n/home05/sqin/OLMo-core/slurm_out/syn-gen-%A_%a.out
#SBATCH -e /n/home05/sqin/OLMo-core/slurm_out/syn-gen-%A_%a.out

set -eo pipefail

MODEL_SIZE=${MODEL_SIZE:?MODEL_SIZE must be set, e.g. MODEL_SIZE=30M}
CHIN=${CHIN:?CHIN must be set, e.g. CHIN=0.05}
NUM_SHARDS=${NUM_SHARDS:-1}
# In array mode, prefer SLURM_ARRAY_TASK_ID. Explicit SHARD_ID still wins
# so we can re-run a single failed shard via plain (non-array) sbatch.
SHARD_ID=${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}

echo "Job started at $(date)"
echo "Host: $(hostname)  SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "MODEL_SIZE=${MODEL_SIZE}  CHIN=${CHIN}  NUM_SHARDS=${NUM_SHARDS}  SHARD_ID=${SHARD_ID}"
echo ""

module purge
module load Mambaforge
module load cuda cudnn
mamba activate openrlhf

cd /n/home05/sqin/OLMo-core/experiment_scripts/self-distill

python -u run_generation.py \
    --model-size "$MODEL_SIZE" \
    --chinchilla-multiplier "$CHIN" \
    --num-shards "$NUM_SHARDS" \
    --shard-id "$SHARD_ID"

echo ""
echo "Job finished at $(date)"
