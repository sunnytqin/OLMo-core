#!/bin/bash

# -----------------------------------------------------------------------------
# Full paraphrase run over dolma train_7.4B.npy (190M chin=2, also covers all
# 30M shards up to train_2.4B by the nested-prefix property).
#
# Default: kempner A100 partition, 32-way SLURM array, 7-day cap per task.
# Switch to H100 by overriding on the command line:
#     sbatch -p kempner_h100 -t 3-00:00 submit_paraphrase_dolma_7.4B.sh
# Array tasks that already made progress on A100 will pick up from their JSONL
# checkpoints via --resume — no loss.
#
# Shape: 4,866,732 docs / 32 shards ≈ 152K docs/task. At ~1 doc/s on A100 that
# is ~42 hrs/task. At H100 ~3 doc/s that's ~14 hrs/task.
# -----------------------------------------------------------------------------

#SBATCH -c 16
#SBATCH -t 7-00:00
#SBATCH -p kempner
#SBATCH --mem=64G
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-31
#SBATCH --exclude=holygpu8a19301,holygpu8a19104,holygpu8a19305,holygpu8a19405
#SBATCH -o /n/home05/sqin/OLMo-core/slurm_out/paraphrase-7.4B-%A_%a.out
#SBATCH -e /n/home05/sqin/OLMo-core/slurm_out/paraphrase-7.4B-%A_%a.out
#SBATCH --account=kempner_dam_lab
#SBATCH --job-name=para-7.4B

module purge
module load cuda cudnn

# Use the env's python interpreter directly — mamba activate is flaky on
# some A100 nodes and silently falls back to system python without numpy.
ENV_PYTHON="/n/holylabs/dam_lab/Lab/sqin/envs/openrlhf/bin/python"
export PATH="/n/holylabs/dam_lab/Lab/sqin/envs/openrlhf/bin:$PATH"

cd /n/home05/sqin/OLMo-core

INPUT="/n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/resharded/allenai/dolma2-tokenizer/train_7.4B.npy"
OUTPUT_DIR="/n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/resharded/allenai/dolma2-tokenizer/paraphrased/train_7.4B_seed0"
NUM_SHARDS=32
SHARD_ID=$SLURM_ARRAY_TASK_ID
SEED=0

echo "=========================================="
echo "Paraphrase: dolma train_7.4B.npy (seed $SEED)"
echo "  Input       : $INPUT"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Shard       : $SHARD_ID / $NUM_SHARDS"
echo "  Seed        : $SEED"
echo "  Partition   : $SLURM_JOB_PARTITION"
echo "  Node        : $(hostname)"
echo "  Job         : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  Started at  : $(date)"
echo "=========================================="
echo ""

$ENV_PYTHON experiment_scripts/paraphrasing/paraphrase_shard.py \
    --input "$INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --shard-id "$SHARD_ID" \
    --num-shards "$NUM_SHARDS" \
    --subsample 1 \
    --seed "$SEED" \
    --temperature 1.0 \
    --top-p 0.95 \
    --max-model-len 16384 \
    --batch-size 500 \
    --resume

echo ""
echo "Finished at $(date)"
