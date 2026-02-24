#!/bin/bash

#SBATCH --job-name=scale-case4  # Override with --job-name on the sbatch command line
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-20:00
#SBATCH --mem=250G
#SBATCH -o ../slurm_out/slurm-%j_%a.out # Standard out goes to this file (%a = array task ID)
#SBATCH -e ../slurm_out/slurm-%j_%a.out # Standard err goes to this file

# Job array for hyperparameter sweep (uncomment to enable, or pass --array=0-9 when submitting)
#SBATCH --array=0-11

# Define parameter grids for job array sweeps
WD_VALUES=(0.1 0.2 0.4)
LR_VALUES=(1e-4 3e-4 1e-3 3e-3)

# Set hyperparameters
SEED=${SEED:-42}
MODEL_SIZE=${MODEL_SIZE:-30M}  # Model size: 30M, 60M, or 370M

# If running as job array, compute WD and LR from SLURM_ARRAY_TASK_ID
# Otherwise use provided values or defaults
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    NUM_LR=${#LR_VALUES[@]}
    WD_IDX=$((SLURM_ARRAY_TASK_ID / NUM_LR))
    LR_IDX=$((SLURM_ARRAY_TASK_ID % NUM_LR))
    WD=${WD_VALUES[$WD_IDX]}
    LR=${LR_VALUES[$LR_IDX]}
else
    LR=${LR:-1e-3}
    WD=${WD:-0.1}
fi
CHIN=${CHIN:-1}  # Chinchilla multiplier
EPOCHS=${EPOCHS:-1}  # Number of epochs (for multi-epoch training)
MICROBATCH_MULT=${MICROBATCH_MULT:-16}  # Microbatch multiplier (rank_microbatch_size = MICROBATCH_MULT * 4096) 8 for A100, 16 for H100 and 32 for H200
NGPU=1  # Number of GPUs per node

echo "Starting training with model_size: $MODEL_SIZE, seed: $SEED, lr: $LR, weight_decay: $WD, chinchilla_multiplier: $CHIN, epochs: $EPOCHS, microbatch_multiplier: $MICROBATCH_MULT"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Load environment
module purge
module load Mambaforge
module load cuda cudnn
mamba activate openrlhf

cd /n/home05/sqin/OLMo-core/

# Create logs directory if it doesn't exist
mkdir -p logs

# Set paths
export WANDB_API_KEY=ffd83b905980a40e959e79930c8a1eb1584f31b9
DATA_ROOT="/n/netscratch/barak_lab/Everyone/sqin/olmo"
WORK_DIR="/n/netscratch/dam_lab/Lab/sqin/olmo/dataset-cache"

# Case 4: Extended DCLM baseline using dolma resharded data (multi-scale)
TRAINING_SCRIPT="src/scripts/official/OLMo-scale-train-case4-dclm-extended.py"
SAVE_FOLDER="/n/netscratch/barak_lab/Lab/sqin/olmo/checkpoints/chinchilla_${CHIN}/${MODEL_SIZE}_seed${SEED}_case4_dolma_epoch${EPOCHS}_wd${WD}_lr${LR}"
echo "Running Case 4: Extended DCLM / dolma (model_size=${MODEL_SIZE}, chinchilla_${CHIN}, ${EPOCHS} epoch(s))"

# Create checkpoint directory
mkdir -p $SAVE_FOLDER

# Use random port to avoid conflicts in multi-user environment
MASTER_PORT=$(shuf -i 29500-65000 -n 1)
echo "Using master port: $MASTER_PORT"

# Enable better CUDA error messages for debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Run training
torchrun --nproc-per-node=$NGPU --master-port=$MASTER_PORT $TRAINING_SCRIPT \
    --save-folder=$SAVE_FOLDER \
    --work-dir=$WORK_DIR \
    --data-root=$DATA_ROOT \
    init_seed=$SEED \
    lr=$LR \
    weight_decay=$WD \
    chinchilla_multiplier=$CHIN \
    epochs=$EPOCHS \
    model_size=$MODEL_SIZE \
    microbatch_multiplier=$MICROBATCH_MULT
