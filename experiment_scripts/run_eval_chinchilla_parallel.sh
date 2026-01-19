#!/bin/bash

#SBATCH -c 1
#SBATCH -t 0-02:00
#SBATCH -p kempner_requeue
#SBATCH --mem=80G
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o ../slurm_out/slurm-chin8-%A_%a.out
#SBATCH -e ../slurm_out/slurm-chin8-%A_%a.out
#SBATCH --account=kempner_barak_lab
#SBATCH --job-name=eval-chin16
#SBATCH --array=0-15  # 16 checkpoints need evaluation

module purge
module load Mambaforge
module load cuda cudnn
mamba activate openrlhf

CHECKPOINT_DIR="/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/chinchilla_4"
STEP="step1145"
OUTPUT_DIR="../results/eval_chinchilla4_individual"

# Create output directory for individual results
mkdir -p "$OUTPUT_DIR"

# List of checkpoints that need evaluation (16 total - case2_dclm_synthetic runs from chinchilla_4)
CHECKPOINTS=(
    "30M_seed42_case2_dclm_synthetic_wd0.1_lr1e-2"
    "30M_seed42_case2_dclm_synthetic_wd0.1_lr1e-3"
    "30M_seed42_case2_dclm_synthetic_wd0.1_lr3e-3"
    "30M_seed42_case2_dclm_synthetic_wd0.1_lr6e-3"
    "30M_seed42_case2_dclm_synthetic_wd0.4_lr1e-2"
    "30M_seed42_case2_dclm_synthetic_wd0.4_lr1e-3"
    "30M_seed42_case2_dclm_synthetic_wd0.4_lr3e-3"
    "30M_seed42_case2_dclm_synthetic_wd0.4_lr6e-3"
    "30M_seed42_case2_dclm_synthetic_wd0.8_lr1e-2"
    "30M_seed42_case2_dclm_synthetic_wd0.8_lr1e-3"
    "30M_seed42_case2_dclm_synthetic_wd0.8_lr3e-3"
    "30M_seed42_case2_dclm_synthetic_wd0.8_lr6e-3"
    "30M_seed42_case2_dclm_synthetic_wd1.6_lr1e-2"
    "30M_seed42_case2_dclm_synthetic_wd1.6_lr1e-3"
    "30M_seed42_case2_dclm_synthetic_wd1.6_lr3e-3"
    "30M_seed42_case2_dclm_synthetic_wd1.6_lr6e-3"
)

# Get the checkpoint for this array task
RUN_NAME="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$RUN_NAME" ]; then
    echo "ERROR: No checkpoint found for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

CHECKPOINT_PATH="$CHECKPOINT_DIR/$RUN_NAME/$STEP"
OUTPUT_FILE="$OUTPUT_DIR/${RUN_NAME}.json"

echo "============================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing: $RUN_NAME"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output: $OUTPUT_FILE"
echo "============================================"
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# Run evaluation - write to individual file
python eval_checkpoints_proper.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --run_name "$RUN_NAME" \
    --output "$OUTPUT_FILE" \
    --sequence_length 4096

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "⚠ Warning: Evaluation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "✓ Done! Results saved to: $OUTPUT_FILE"
