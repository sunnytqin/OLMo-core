#!/bin/bash

#SBATCH --job-name=eval-scaling
#SBATCH --account=kempner_barak_lab
#SBATCH --partition=kempner_requeue
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem=250G
#SBATCH -o ../slurm_out/slurm-eval-scaling-%A_%a.out
#SBATCH -e ../slurm_out/slurm-eval-scaling-%A_%a.out
#SBATCH --array=0-11

module purge
module load Mambaforge
module load cuda cudnn
mamba activate openrlhf

cd /n/home05/sqin/OLMo-core/experiment_scripts/

python run_eval_batch.py \
    --manifest /n/home05/sqin/OLMo-core/results/chinchilla_fit_dolma/manifest_needs_eval.json \
    --results-dir /n/home05/sqin/OLMo-core/results/dolma_val_loss \
    --worker-id $SLURM_ARRAY_TASK_ID \
    --num-workers 12

echo "Worker $SLURM_ARRAY_TASK_ID done."
