#!/bin/bash

#SBATCH -c 16
#SBATCH -t 0-24:00
#SBATCH -p kempner_h100
#SBATCH --mem=64G
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-15
#SBATCH -o /n/home05/sqin/OLMo-core/slurm_out/paraphrase-%A_%a.out
#SBATCH -e /n/home05/sqin/OLMo-core/slurm_out/paraphrase-%A_%a.out
#SBATCH --account=kempner_dam_lab
#SBATCH --job-name=paraphrase

module purge
module load Mambaforge
module load cuda cudnn
mamba activate openrlhf

cd /n/home05/sqin/OLMo-core

INPUT="/n/netscratch/dam_lab/Lab/sqin/olmo/preprocessed/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/resharded/train_0.3B.npy"
OUTPUT_DIR="/n/netscratch/dam_lab/Lab/sqin/olmo/preprocessed/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/paraphrased/train_0.3B_subsample_2"
NUM_SHARDS=16
SHARD_ID=$SLURM_ARRAY_TASK_ID

echo "Job started at $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Shard: $SHARD_ID / $NUM_SHARDS"
echo ""

python experiment_scripts/paraphrasing/paraphrase_shard.py \
    --input "$INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --shard-id $SHARD_ID \
    --num-shards $NUM_SHARDS \
    --max-model-len 16384 \
    --batch-size 500 \
    --resume

echo ""
echo "Job finished at $(date)"
