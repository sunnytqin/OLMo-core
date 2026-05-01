#!/bin/bash

#SBATCH --job-name=scale-paraphrase  # Override with --job-name on the sbatch command line
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-20:00
#SBATCH --mem=250G
#SBATCH -o ../slurm_out/slurm-%j_%a.out
#SBATCH -e ../slurm_out/slurm-%j_%a.out

# Job array for hyperparameter sweep (pass --array=0-N when submitting to enable)
##SBATCH --array=0-1

# Define parameter grids for job array sweeps
WD_VALUES=(0.1 0.2)
LR_VALUES=(3e-3)

# Set hyperparameters
SEED=${SEED:-42}
MODEL_SIZE=${MODEL_SIZE:-30M}  # 14M, 30M, 60M, 100M, 190M, 370M, 600M

# Array sweep over WD x LR if SLURM_ARRAY_TASK_ID is set
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

CHIN=${CHIN:-1}                   # Chinchilla multiplier — sets D's size
NUM_SEEDS=${NUM_SEEDS:-1}         # K = number of paraphrase seeds (1..8); D' = K copies of paraphrased D
MICROBATCH_MULT=${MICROBATCH_MULT:-16}  # 8=A100, 16=H100, 32=H200
EVAL_ONLY=${EVAL_ONLY:-false}
NGPU=${NGPU:-1}

echo "Starting paraphrase training with model_size=${MODEL_SIZE}, seed=${SEED}, lr=${LR}, wd=${WD}, chinchilla=${CHIN}, num_seeds=${NUM_SEEDS}, microbatch_mult=${MICROBATCH_MULT}"
echo "Job ID: $SLURM_JOB_ID  Host: $(hostname)  GPUs: $CUDA_VISIBLE_DEVICES"

# Environment
module purge
module load Mambaforge
module load cuda cudnn
mamba activate openrlhf

cd /n/home05/sqin/OLMo-core/
mkdir -p logs

export WANDB_API_KEY=ffd83b905980a40e959e79930c8a1eb1584f31b9
DATA_ROOT="/n/netscratch/barak_lab/Everyone/sqin/olmo"
WORK_DIR="/n/netscratch/dam_lab/Lab/sqin/olmo/dataset-cache"

# Paraphrase training: D + D' for 1 epoch, K paraphrase seeds
TRAINING_SCRIPT="src/scripts/official/OLMo-scale-train-paraphrase-dolma.py"
SAVE_FOLDER="/n/netscratch/barak_lab/Lab/sqin/olmo/checkpoints/chinchilla_${CHIN}/${MODEL_SIZE}_seed${SEED}_dolma_para_K${NUM_SEEDS}_wd${WD}_lr${LR}"
echo "Paraphrase D+D' (model=${MODEL_SIZE}, chin=${CHIN}, K=${NUM_SEEDS})"

mkdir -p $SAVE_FOLDER

MASTER_PORT=$(shuf -i 29500-65000 -n 1)
echo "Master port: $MASTER_PORT"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

torchrun --nproc-per-node=$NGPU --master-port=$MASTER_PORT $TRAINING_SCRIPT \
    --save-folder=$SAVE_FOLDER \
    --work-dir=$WORK_DIR \
    --data-root=$DATA_ROOT \
    init_seed=$SEED \
    lr=$LR \
    weight_decay=$WD \
    chinchilla_multiplier=$CHIN \
    num_seeds=$NUM_SEEDS \
    model_size=$MODEL_SIZE \
    microbatch_multiplier=$MICROBATCH_MULT \
    eval_only=$EVAL_ONLY
