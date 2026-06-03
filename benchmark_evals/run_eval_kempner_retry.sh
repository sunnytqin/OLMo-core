#!/bin/bash

#SBATCH --job-name=bench-eval-370m
#SBATCH --account=kempner_barak_lab
#SBATCH --partition=kempner
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00
#SBATCH --mem=64G
#SBATCH -o /n/home05/sqin/OLMo-core/slurm_out/bench-eval-370m-%A_%a.out
#SBATCH -e /n/home05/sqin/OLMo-core/slurm_out/bench-eval-370m-%A_%a.out
#SBATCH --array=0-1

set -euo pipefail

module purge
module load Mambaforge
module load cuda cudnn

PYTHON=/n/holylabs/dam_lab/Lab/sqin/envs/openrlhf/bin/python

BEST_RUNS=/n/netscratch/barak_lab/Lab/sqin/olmo/checkpoints/chinchilla_0.1/best_runs

CHECKPOINTS=(
  "$BEST_RUNS/370M_seed42_multiepoch_dolma_chin0.1_epoch4_wd0.8_lr0.001/step1412"
  "$BEST_RUNS/370M_seed42_multiepoch_dolma_chin0.1_epoch8_wd0.2_lr0.003/step2823"
)

CKPT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
echo "Worker $SLURM_ARRAY_TASK_ID -> $CKPT"

HF_DIR=/tmp/hf_models_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$HF_DIR"
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$TRITON_CACHE_DIR"

cd /n/home05/sqin/OLMo-core

"$PYTHON" -u benchmark_evals/convert_and_eval.py \
  --checkpoint "$CKPT" \
  --limit 1000 \
  --output-dir "$HF_DIR" \
  --results-dir /n/home05/sqin/OLMo-core/benchmark_evals/results

echo "Worker $SLURM_ARRAY_TASK_ID done."
