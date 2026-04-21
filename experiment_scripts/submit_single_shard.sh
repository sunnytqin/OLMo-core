#!/bin/bash

#SBATCH -c 4
#SBATCH -t 0-12:00
#SBATCH -p test
#SBATCH --mem=128G
#SBATCH -n 1
#SBATCH -o ../slurm_out/doc_splits-%j.out
#SBATCH -e ../slurm_out/doc_splits-%j.out
#SBATCH --account=dam_lab
#SBATCH --job-name=doc-splits

# Usage: sbatch --export=ALL,ONLY=train_15.2B submit_single_shard.sh

module purge
module load Mambaforge
mamba activate openrlhf

cd /n/home05/sqin/OLMo-core

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Shard filter: ${ONLY}"
echo ""

python experiment_scripts/create_dolma3_splits.py \
    --data-dir /n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/v0.1-150b/allenai/dolma2-tokenizer \
    --output-dir /n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/resharded/allenai/dolma2-tokenizer \
    --seed 42 \
    --only "${ONLY}"

echo ""
echo "Job finished at $(date)"
