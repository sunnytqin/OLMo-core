#!/bin/bash
#SBATCH --job-name=rebuild-7.4B
#SBATCH --partition=sapphire
#SBATCH --account=barak_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH -o ../slurm_out/rebuild-7.4B-%j.out
#SBATCH -e ../slurm_out/rebuild-7.4B-%j.out
set -u
PY=/n/holylabs/dam_lab/Lab/sqin/envs/openrlhf/bin/python
DATA_DIR=/n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/v0.1-150b/allenai/dolma2-tokenizer
OUT_DIR=/n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/resharded/allenai/dolma2-tokenizer
cd /n/home05/sqin/OLMo-core
echo "[$(date +%F_%T)] host=$(hostname) rebuilding train_7.4B ONLY (cached doc_index)"
$PY experiment_scripts/create_dolma3_splits.py \
    --data-dir "$DATA_DIR" --output-dir "$OUT_DIR" --seed 42 \
    --only train_7.4B --overwrite
echo "[$(date +%F_%T)] exited $?  final size:"
ls -lh "$OUT_DIR/train_7.4B.npy"
