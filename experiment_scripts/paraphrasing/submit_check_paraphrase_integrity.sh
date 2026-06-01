#!/bin/bash
# -----------------------------------------------------------------------------
# Integrity check for all paraphrased seed dirs. CPU-bound (memmap reads + EOS
# counting), but submitted on a GPU partition to match the cluster norms.
#
# Submit with:
#   sbatch -p gpu_test experiment_scripts/paraphrasing/submit_check_paraphrase_integrity.sh
#   sbatch -p kempner --account=kempner_barak_lab \
#       experiment_scripts/paraphrasing/submit_check_paraphrase_integrity.sh
#
# Active running tasks (passed via env var SKIP) are reported but not failed.
# -----------------------------------------------------------------------------

#SBATCH -c 4
#SBATCH -t 1:00:00
#SBATCH --mem=16G
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o /n/home05/sqin/OLMo-core/slurm_out/check_paraphrase_integrity-%j.out
#SBATCH -e /n/home05/sqin/OLMo-core/slurm_out/check_paraphrase_integrity-%j.out
#SBATCH --job-name=para-check

set -euo pipefail

ENV_PYTHON="/n/holylabs/dam_lab/Lab/sqin/envs/openrlhf/bin/python"
cd /n/home05/sqin/OLMo-core

SEEDS="${SEEDS:-1-32}"
SKIP="${SKIP:-}"

echo "=========================================="
echo "Integrity check for paraphrased seeds"
echo "  Seeds        : $SEEDS"
echo "  Skip pairs   : ${SKIP:-(none)}"
echo "  Partition    : $SLURM_JOB_PARTITION"
echo "  Node         : $(hostname)"
echo "  Job          : $SLURM_JOB_ID"
echo "  Started at   : $(date)"
echo "=========================================="

"$ENV_PYTHON" -u experiment_scripts/paraphrasing/check_paraphrase_integrity.py \
    --seeds "$SEEDS" \
    --skip "$SKIP"

echo ""
echo "Finished at $(date)"
