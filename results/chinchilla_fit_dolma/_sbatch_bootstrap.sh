#!/bin/bash
#SBATCH --job-name=bootstrap_ci
#SBATCH --output=/home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma/_slurm_logs/bootstrap_%j.out
#SBATCH --error=/home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma/_slurm_logs/bootstrap_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=4gb

set -e
mkdir -p /home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma/_slurm_logs
mkdir -p /home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma/_onego_json
cd /home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma

VARIANT="$1"
B="${2:-200}"
SEED="${3:-42}"

ANCHOR="_onego_json/onego_${VARIANT}_para_dense.json"
OUT="_onego_json/bootstrap_${VARIANT}_B${B}_seed${SEED}.json"

echo "Job $SLURM_JOB_ID  variant=$VARIANT  B=$B  seed=$SEED"
echo "Anchor: $ANCHOR"
echo "Out:    $OUT"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"

PYTHONUNBUFFERED=1 /home/hamidieh/anaconda3/bin/python \
    bootstrap_onego.py \
    --anchor-json "$ANCHOR" \
    --variant "$VARIANT" \
    --B "$B" \
    --seed "$SEED" \
    --out-json "$OUT"

echo "Finished: $(date)"
