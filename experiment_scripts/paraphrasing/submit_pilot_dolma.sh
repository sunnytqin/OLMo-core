#!/bin/bash

#SBATCH -c 16
#SBATCH -t 0-02:00
#SBATCH -p kempner
#SBATCH --mem=64G
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --exclude=holygpu8a19301,holygpu8a19104,holygpu8a19305,holygpu8a19405
#SBATCH -o /n/home05/sqin/OLMo-core/slurm_out/paraphrase-pilot-%j.out
#SBATCH -e /n/home05/sqin/OLMo-core/slurm_out/paraphrase-pilot-%j.out
#SBATCH --account=kempner_dam_lab
#SBATCH --job-name=para-pilot

# Pilot: paraphrase a small slice (~1000 docs) of dolma train_0.03B.npy to
# sanity-check the new length-preserving Wikipedia prompt and measure
# throughput before committing the full 7.4B run.
#
# train_0.03B.npy has ~35K docs; shard 0 of 35 shards ≈ 1000 docs.

module purge
module load cuda cudnn

# Use the env's python interpreter directly — `mamba activate openrlhf`
# is flaky on some A100 nodes and silently falls back to system python
# without numpy. Absolute path is robust.
ENV_PYTHON="/n/holylabs/dam_lab/Lab/sqin/envs/openrlhf/bin/python"
export PATH="/n/holylabs/dam_lab/Lab/sqin/envs/openrlhf/bin:$PATH"

cd /n/home05/sqin/OLMo-core

INPUT="/n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/resharded/allenai/dolma2-tokenizer/train_0.03B.npy"
OUTPUT_DIR="/n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/resharded/allenai/dolma2-tokenizer/paraphrased_pilot/train_0.03B_seed0_v3"
NUM_SHARDS=35
SHARD_ID=0
SEED=0

echo "Pilot paraphrase run"
echo "  Input      : $INPUT"
echo "  Output dir : $OUTPUT_DIR"
echo "  Shard      : $SHARD_ID / $NUM_SHARDS"
echo "  Seed       : $SEED"
echo "  Started at : $(date)"
echo "  Host       : $(hostname)"
echo ""

$ENV_PYTHON experiment_scripts/paraphrasing/paraphrase_shard.py \
    --input "$INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --shard-id $SHARD_ID \
    --num-shards $NUM_SHARDS \
    --subsample 1 \
    --seed $SEED \
    --temperature 1.0 \
    --top-p 0.95 \
    --max-model-len 16384 \
    --batch-size 500 \
    --resume

echo ""
echo "Pilot finished at $(date)"
echo ""
echo "Next: run inspection script to check quality and length preservation:"
echo "  python experiment_scripts/paraphrasing/inspect_pilot.py \\"
echo "      --input $INPUT \\"
echo "      --checkpoint $OUTPUT_DIR/shard_0000.jsonl \\"
echo "      --num-shards $NUM_SHARDS --shard-id $SHARD_ID --subsample 1"
