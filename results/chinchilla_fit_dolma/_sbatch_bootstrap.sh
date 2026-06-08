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

MODE="$1"            # one of: variant, xval
ARG="$2"             # variant name (all/drop14m) OR n_max_mil cutoff
EXTRA="$3"           # for xval: "anchored" or empty
B="${4:-200}"
SEED="${5:-42}"

if [ "$MODE" = "variant" ]; then
    ANCHOR="_onego_json/onego_${ARG}_default.json"
    OUT="_onego_json/bootstrap_${ARG}_B${B}_seed${SEED}.json"
    EXTRA_FLAGS="--variant $ARG"
elif [ "$MODE" = "xval" ]; then
    SUFFIX=""
    if [ "$EXTRA" = "anchored" ]; then SUFFIX="_anchored"; fi
    ANCHOR="_xval_json/xval_cut${ARG}${SUFFIX}.json"
    OUT="_xval_json/bootstrap_xval_cut${ARG}${SUFFIX}_B${B}_seed${SEED}.json"
    EXTRA_FLAGS="--variant all --n-max-mil $ARG"
else
    echo "ERROR: MODE must be variant or xval" >&2; exit 1
fi

echo "Job $SLURM_JOB_ID  mode=$MODE  arg=$ARG  extra=$EXTRA  B=$B  seed=$SEED"
echo "Anchor: $ANCHOR"
echo "Out:    $OUT"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"

PYTHONUNBUFFERED=1 /home/hamidieh/anaconda3/bin/python \
    bootstrap_onego.py \
    --anchor-json "$ANCHOR" \
    $EXTRA_FLAGS \
    --B "$B" \
    --seed "$SEED" \
    --out-json "$OUT"

echo "Finished: $(date)"
