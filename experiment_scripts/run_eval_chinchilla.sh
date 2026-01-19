#!/bin/bash

#SBATCH -c 1 # Number of cores requested
#SBATCH -t 0-12:00 # Runtime in minutes
#SBATCH -p kempner_requeue # Partition to submit to
#SBATCH --mem=80G
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o ../slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e ../slurm_out/slurm-%j.out # Standard err goes to this filehostname hostname
#SBATCH --account=kempner_dam_lab
#SBATCH --job-name=e-chin8


module purge
module load Mambaforge
module load cuda cudnn
mamba activate openrlhf



# Script to evaluate all checkpoints in chinchilla_x
# Each checkpoint is evaluated in a separate Python process to avoid GPU OOM

CHECKPOINT_DIR="/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/chinchilla_8"
STEP="step1145"
# STEP="step2289"
# STEP="step4578"

OUTPUT="../results/eval_results_chinchilla8.json"

echo "Evaluating checkpoints in: $CHECKPOINT_DIR"
echo "Checkpoint step: $STEP"
echo "Output file: $OUTPUT"
echo ""

# Count total checkpoints
TOTAL=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "30M_*" | wc -l)
echo "Found $TOTAL checkpoints to evaluate"
echo ""

# Loop through each run directory
COUNT=0
for RUN_DIR in "$CHECKPOINT_DIR"/30M_*; do
    if [ ! -d "$RUN_DIR" ]; then
        continue
    fi

    COUNT=$((COUNT + 1))
    RUN_NAME=$(basename "$RUN_DIR")
    CHECKPOINT_PATH="$RUN_DIR/$STEP"

    echo "[$COUNT/$TOTAL] Processing: $RUN_NAME"

    # Run evaluation in a separate Python process (exits after completion, freeing GPU memory)
    python eval_checkpoints_proper.py \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --run_name "$RUN_NAME" \
        --output "$OUTPUT" \
        --sequence_length 4096

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "⚠ Warning: Evaluation failed with exit code $EXIT_CODE"
    fi

    echo ""
done

echo "Done! Results saved to: $OUTPUT"

# Print summary
echo ""
echo "SUMMARY"
echo "========================================"
if [ -f "$OUTPUT" ]; then
    python -c "
import json
with open('$OUTPUT', 'r') as f:
    results = json.load(f)
print(f'Total runs evaluated: {len(results)}')
print(f'{\"Run\":<60} {\"Val Loss\":>10} {\"PPL\":>10}')
print('-' * 81)
for run_name in sorted(results.keys()):
    result = results[run_name]
    if 'error' in result:
        print(f'{run_name:<60} {\"ERROR\":>10} {\"ERROR\":>10}')
    else:
        loss = result['validation_loss']
        ppl = result['perplexity']
        print(f'{run_name:<60} {loss:>10.4f} {ppl:>10.2f}')
"
else
    echo "No results file found"
fi
