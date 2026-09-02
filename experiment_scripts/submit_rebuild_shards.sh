#!/bin/bash
#SBATCH --job-name=rebuild-dolma-shards
#SBATCH --partition=sapphire
#SBATCH --account=barak_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00
#SBATCH -o ../slurm_out/rebuild-shards-%j.out
#SBATCH -e ../slurm_out/rebuild-shards-%j.out

# Rebuild the 18 dolma D shards after the netscratch 90-day purge.
# Deterministic (--seed 42) reconstruction from the re-downloaded source corpus.
# Verified afterward against the surviving shard_manifest.json (exact token+doc counts).

set -u
PY=/n/holylabs/dam_lab/Lab/sqin/envs/openrlhf/bin/python
DATA_DIR=/n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/v0.1-150b/allenai/dolma2-tokenizer
OUT_DIR=/n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/resharded/allenai/dolma2-tokenizer

SHARDS=train_0.03B,train_0.06B,train_0.15B,train_0.6B,train_1.2B,train_2.4B,train_4.8B,train_0.12B,train_0.3B,train_0.19B,train_0.38B,train_0.95B,train_1.9B,train_3.8B,train_7.4B,train_0.37B,train_0.74B,train_3.7B

cd /n/home05/sqin/OLMo-core

echo "[$(date +%F_%T)] host=$(hostname)  rebuilding 18 shards, seed=42"
$PY experiment_scripts/create_dolma3_splits.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT_DIR" \
    --seed 42 \
    --only "$SHARDS" \
    --overwrite

echo "[$(date +%F_%T)] create_dolma3_splits exited $?"
