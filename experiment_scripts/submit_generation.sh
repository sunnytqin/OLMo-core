#!/bin/bash

#SBATCH -c 16                          # Number of CPU cores
#SBATCH -t 0-12:00                     # Runtime (48 hours)
#SBATCH -p kempner_h100                # Partition to submit to
#SBATCH --mem=64G                      # Memory
#SBATCH -n 1                           # Number of nodes
#SBATCH --gres=gpu:1                   # Number of GPUs per task
#SBATCH --array=0-63                  # Job array: 64 shards (0-63)
#SBATCH -o ../slurm_out/generate-%A_%a.out # Standard out (%A=job ID, %a=array index)
#SBATCH -e ../slurm_out/generate-%A_%a.out # Standard err
#SBATCH --account=kempner_dam_lab
#SBATCH --job-name=syn-gen


module purge
module load Mambaforge
module load cuda cudnn
mamba activate openrlhf

cd /n/home05/sqin/OLMo-core/experiment_scripts

# Configuration
MODEL_PATH="/n/netscratch/barak_lab/Lab/sqin/olmo/checkpoints/chinchilla_0.5/30M_seed42_case4_dclm_extended_epoch64_wd0.1_lr1e-2/step9156_hf"
OUTPUT_DIR="/n/netscratch/dam_lab/Lab/sqin/olmo/generated_data/30M_chinchilla0.5_temp1.0"
TOTAL_TOKENS=19200000000  # 19.2B tokens total (0.3B per shard)
NUM_SHARDS=64              # Number of parallel shards
SHARD_ID=$SLURM_ARRAY_TASK_ID  # Use array task ID as shard ID
SEQUENCE_LENGTH=4096
BATCH_SIZE=1024
TEMPERATURE=1.0
SEED=42

# Print job info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Shard: $SHARD_ID / $NUM_SHARDS"
echo "SLURM_GPUS: $SLURM_GPUS"
echo ""

# Run generation script
python3 generate_unconditional_vllm.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --total-tokens $TOTAL_TOKENS \
    --sequence-length $SEQUENCE_LENGTH \
    --batch-size $BATCH_SIZE \
    --temperature $TEMPERATURE \
    --seed $SEED \
    --shard-id $SHARD_ID \
    --num-shards $NUM_SHARDS \
    --resume

echo ""
echo "Job finished at $(date)"

