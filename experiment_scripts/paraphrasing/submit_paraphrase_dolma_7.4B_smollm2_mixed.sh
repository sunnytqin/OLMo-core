#!/bin/bash

# -----------------------------------------------------------------------------
# Full paraphrase run over dolma train_7.4B.npy with SmolLM2-1.7B-Instruct +
# suffix speculative decoding + per-doc mixed-prompt sampling (faq, math,
# table, tutorial). Seed 2.
#
# Output: paraphrased/train_7.4B_smollm2_mixed_seed2/shard_{0000..0031}.{jsonl,npy}
#
# Default: kempner_h100, 3-day cap. Override partition at submit time:
#   H200:  sbatch -p gpu_h200 -t 3-00:00 --account=barak_lab \
#          --exclude=holygpu8a12501 --array=<range> <this script>
#   A100:  sbatch -p kempner -t 7-00:00 --account=kempner_dam_lab \
#          --exclude=holygpu8a19301,holygpu8a19104,holygpu8a19305,holygpu8a19405 \
#          --array=<range> <this script>
#
# JSONL checkpoints preserve progress across partition migration via --resume.
# -----------------------------------------------------------------------------

#SBATCH -c 16
#SBATCH -t 3-00:00
#SBATCH -p kempner_h100
#SBATCH --mem=64G
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-31
#SBATCH -o /n/home05/sqin/OLMo-core/slurm_out/paraphrase-7.4B-s2sm-%A_%a.out
#SBATCH -e /n/home05/sqin/OLMo-core/slurm_out/paraphrase-7.4B-s2sm-%A_%a.out
#SBATCH --account=kempner_dam_lab
#SBATCH --job-name=para-s2sm

module purge
module load cuda cudnn

ENV_PYTHON="/n/holylabs/dam_lab/Lab/sqin/envs/openrlhf/bin/python"
export PATH="/n/holylabs/dam_lab/Lab/sqin/envs/openrlhf/bin:$PATH"

cd /n/home05/sqin/OLMo-core

INPUT="/n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/resharded/allenai/dolma2-tokenizer/train_7.4B.npy"
# SEED is picked up from env (override via `sbatch --export=PARAPHRASE_SEED=N,ALL`);
# default matches the original SmolLM2 mixed run.
SEED=${PARAPHRASE_SEED:-2}
OUTPUT_DIR="/n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/resharded/allenai/dolma2-tokenizer/paraphrased/train_7.4B_smollm2_mixed_seed${SEED}"
NUM_SHARDS=32
SHARD_ID=$SLURM_ARRAY_TASK_ID

echo "=========================================="
echo "Paraphrase: dolma train_7.4B.npy (SmolLM2, seed $SEED, mixed prompts)"
echo "  Input       : $INPUT"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Shard       : $SHARD_ID / $NUM_SHARDS"
echo "  Seed        : $SEED"
echo "  Prompt style: mixed (faq/math/table/tutorial per-doc)"
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
    --max-model-len 8192 \
    --batch-size 500 \
    --prompt-style mixed \
    --resume

echo ""
echo "Finished at $(date)"
