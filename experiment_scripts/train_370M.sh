#!/bin/bash

#SBATCH --job-name=2-370M-16
#SBATCH --account=kempner_dam_lab 
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-06:00
#SBATCH --mem=250G
#SBATCH -o ../slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e ../slurm_out/slurm-%j.out # Standard err goes to this file

# Set hyperparameters (can be overridden when submitting: sbatch --export=SEED=123,LR=1e-3,WD=0.1,CHIN=64,EPOCHS=1,REPEAT=64,NGPU=2 train_30M.sh)
SEED=${SEED:-42}
LR=${LR:-1e-3}
WD=${WD:-0.1}
CHIN=48  # Chinchilla multiplier (e.g., 48 means 48x0.3B=14.4B tokens)
EPOCHS=${EPOCHS:-1}  # Case 4 only, Number of epochs (for multi-epoch training)
REPEAT=${REPEAT:-16}  # Case 2 only, DCLM repeat factor (16 for repeat16-synthetic48)
NGPU=2  # Number of GPUs per node

echo "Starting training with seed: $SEED, lr: $LR, weight_decay: $WD, chinchilla_multiplier: $CHIN, epochs: $EPOCHS, dclm_repeat: $REPEAT"
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
# NOTE: For repeat16-synthetic48 (REPEAT=16): set CHIN=48 for ~14.4B tokens (16×0.3B DCLM + 14.4B synthetic)
# TRAINING_SCRIPT="src/scripts/official/OLMo-tiny-train-case2-dclm-synthetic.py"
# if [ $REPEAT -gt 1 ]; then
#     SAVE_FOLDER="/n/netscratch/barak_lab/Lab/sqin/olmo/syn_data/chinchilla_${CHIN}/370M_seed${SEED}_case2_dclm_synthetic_repeat${REPEAT}_wd${WD}_lr${LR}"
#     echo "Running Case 2: DCLM + Synthetic (chinchilla_${CHIN}, DCLM repeated ${REPEAT}x)"
# else
#     SAVE_FOLDER="/n/netscratch/barak_lab/Lab/sqin/olmo/syn_data/chinchilla_${CHIN}/370M_seed${SEED}_case2_dclm_synthetic_wd${WD}_lr${LR}"
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
    epochs=$EPOCHS

