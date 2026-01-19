#!/bin/bash

#SBATCH --job-name=0.5-16e-30M
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-02:00
#SBATCH --mem=250G
#SBATCH -o ../slurm_out/slurm-%j_%a.out # Standard out goes to this file (%a = array task ID)
#SBATCH -e ../slurm_out/slurm-%j_%a.out # Standard err goes to this file

# Job array for hyperparameter sweep (uncomment to enable, or pass --array=0-9 when submitting)
#SBATCH --array=0-9

# Define parameter grids for job array sweeps
WD_VALUES=(0.1 0.4)
LR_VALUES=(3e-3 6e-3 1e-2 3e-2 6e-2)

# Set hyperparameters
SEED=${SEED:-42}

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
CHIN=0.5  # Chinchilla multiplier (e.g., 64 means 64x0.6B=38.4B tokens)
EPOCHS=${EPOCHS:-1}  # Case 4 only, Number of epochs (for multi-epoch training)
REPEAT=${REPEAT:-32}  # Case 2 only, DCLM repeat factor (32 or 64 for chin64)
MICROBATCH_MULT=${MICROBATCH_MULT:-16}  # Microbatch multiplier (rank_microbatch_size = MICROBATCH_MULT * 4096)
NGPU=2  # Number of GPUs per node

echo "Starting training with seed: $SEED, lr: $LR, weight_decay: $WD, chinchilla_multiplier: $CHIN, epochs: $EPOCHS, dclm_repeat: $REPEAT, microbatch_multiplier: $MICROBATCH_MULT"
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
DATA_ROOT="/n/netscratch/dam_lab/Lab/sqin/olmo"
WORK_DIR="/n/netscratch/dam_lab/Lab/sqin/olmo/dataset-cache"

################################################################################
# EXPERIMENT SELECTION - Uncomment ONE block below to run that case
################################################################################

# Case 2: DCLM + Synthetic ("mocked" multi-epoch on DCLM, single pass on synthetic)
# NOTE: For repeat32-synthetic32 (REPEAT=32): set CHIN=61 for ~36.6B tokens (32×0.6B DCLM + 17.4B synthetic)
# NOTE: For repeat64-synthetic32 (REPEAT=64): set CHIN=93 for ~55.8B tokens (64×0.6B DCLM + 17.4B synthetic)
# TRAINING_SCRIPT="src/scripts/official/OLMo-tiny-train-case2-dclm-synthetic.py"
# if [ $REPEAT -gt 1 ]; then
#     SAVE_FOLDER="/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/chinchilla_${CHIN}/30M_seed${SEED}_case2_dclm_synthetic_repeat${REPEAT}_wd${WD}_lr${LR}"
#     echo "Running Case 2: DCLM + Synthetic (chinchilla_${CHIN}, DCLM repeated ${REPEAT}x)"
# else
#     SAVE_FOLDER="/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/chinchilla_${CHIN}/30M_seed${SEED}_case2_dclm_synthetic_wd${WD}_lr${LR}"
#     echo "Running Case 2: DCLM + Synthetic (chinchilla_${CHIN}, no repetition)"
# fi

# Case 3: DCLM with repetition
# TRAINING_SCRIPT="src/scripts/official/OLMo-tiny-train-case3-dclm-repeat.py"
# SAVE_FOLDER="/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/chinchilla_${CHIN}/370M_seed${SEED}_case3_dclm_repeat_wd${WD}_lr${LR}"
# echo "Running Case 3: DCLM with repetition (chinchilla_${CHIN})"

# Case 4: Extended DCLM (multi-epoch support, dataset selection based on chinchilla_multiplier)
TRAINING_SCRIPT="src/scripts/official/OLMo-tiny-train-case4-dclm-extended.py"
SAVE_FOLDER="/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/chinchilla_${CHIN}/30M_seed${SEED}_case4_dclm_extended_epoch${EPOCHS}_wd${WD}_lr${LR}"
echo "Running Case 4: Extended DCLM (chinchilla_${CHIN}, ${EPOCHS} epoch(s))"

################################################################################

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
    dclm_repeat_factor=$REPEAT \
    epochs=$EPOCHS \
    microbatch_multiplier=$MICROBATCH_MULT

