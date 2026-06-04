#!/bin/bash
#SBATCH --job-name=xval
#SBATCH --output=/home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma/_slurm_logs/xval_%j.out
#SBATCH --error=/home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma/_slurm_logs/xval_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=4gb

set -e
mkdir -p /home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma/_xval_json
cd /home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma

N_MAX_MIL="$1"
GRID="${2:-default}"
ANCHORED="${3:-no}"   # "yes" or "no"
TAG="cut${N_MAX_MIL}_${GRID}"
[ "$ANCHORED" = "yes" ] && TAG="${TAG}_anchored"
OUT_JSON="/home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma/_xval_json/xval_${TAG}.json"

echo "Job $SLURM_JOB_ID  cutoff=${N_MAX_MIL}M  grid=$GRID  anchored=$ANCHORED"
echo "Out: $OUT_JSON"
echo "Started: $(date)"

CMD=(/home/hamidieh/anaconda3/bin/python fit_triple_xval.py \
     --n-max-mil "$N_MAX_MIL" --grid "$GRID" \
     --out-json "$OUT_JSON")
[ "$ANCHORED" = "yes" ] && CMD+=(--anchored)

PYTHONUNBUFFERED=1 "${CMD[@]}"

echo "Finished: $(date)"
