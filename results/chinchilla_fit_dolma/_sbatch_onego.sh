#!/bin/bash
#SBATCH --job-name=onego_fit
#SBATCH --output=/home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma/_slurm_logs/onego_%j.out
#SBATCH --error=/home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma/_slurm_logs/onego_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=4gb

set -e
mkdir -p /home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma/_slurm_logs
mkdir -p /home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma/_onego_json
cd /home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma

VARIANT="$1"
GRID="$2"
TAG="${VARIANT}_${GRID}"
OUT_JSON="/home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma/_onego_json/onego_${TAG}.json"
echo "Job $SLURM_JOB_ID  variant=$VARIANT  grid=$GRID  tag=$TAG"
echo "Node: $SLURMD_NODENAME"
echo "OutJSON: $OUT_JSON"
echo "Started: $(date)"

PYTHONUNBUFFERED=1 /home/hamidieh/anaconda3/bin/python \
    fit_joint_triple_onego.py \
    --variant "$VARIANT" --grid "$GRID" \
    --out-json "$OUT_JSON"

echo "Finished: $(date)"
