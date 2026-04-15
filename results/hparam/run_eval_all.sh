#!/bin/bash

#SBATCH --job-name=eval-batch
#SBATCH --account=kempner_barak_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00
#SBATCH --mem=250G
#SBATCH -o ../slurm_out/slurm-eval-batch-%A_%a.out
#SBATCH -e ../slurm_out/slurm-eval-batch-%A_%a.out
#SBATCH --array=0-7

# Each array task processes ~1/8 of the manifest in-process.
# Model architecture + data loader are loaded once per model size,
# then checkpoint weights are swapped for each run.

module purge
module load Mambaforge
module load cuda cudnn
mamba activate openrlhf

cd /n/home05/sqin/OLMo-core/experiment_scripts/

python run_eval_batch.py \
    --manifest /n/home05/sqin/OLMo-core/results/hparam/manifest.json \
    --results-dir /n/home05/sqin/OLMo-core/results/dolma_val_loss \
    --worker-id $SLURM_ARRAY_TASK_ID \
    --num-workers 8

echo "Worker $SLURM_ARRAY_TASK_ID done."
