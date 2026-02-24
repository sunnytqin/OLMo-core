#!/bin/bash

#SBATCH -c 4                           # CPU cores (scanning is mostly I/O bound)
#SBATCH -t 0-24:00                     # Runtime (4 hours, scan ~1h + writes ~1h)
#SBATCH -p kempner                # Partition
#SBATCH --mem=128G                     # Memory (doc index for 150B tokens ~3GB arrays + buffer)
#SBATCH -n 1                           # Number of nodes
#SBATCH --gres=gpu:1                   # Required by partition (script is CPU-only)
#SBATCH -o ../slurm_out/doc_splits-%j.out
#SBATCH -e ../slurm_out/doc_splits-%j.out
#SBATCH --account=kempner_dam_lab
#SBATCH --job-name=doc-splits

module purge
module load Mambaforge
mamba activate openrlhf

cd /n/home05/sqin/OLMo-core

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo ""

python experiment_scripts/create_dolma3_splits.py \
    --data-dir /n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/v0.1-150b/allenai/dolma2-tokenizer \
    --output-dir /n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/resharded/dolma2-tokenizer \
    --seed 42

echo ""
echo "Job finished at $(date)"
