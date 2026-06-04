#!/bin/bash
#SBATCH --job-name=extrap
#SBATCH --output=/home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma/_slurm_logs/extrap_%j.out
#SBATCH --error=/home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma/_slurm_logs/extrap_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=4gb

set -e
cd /home/hamidieh/projects/syn_pt/OLMo-core/results/chinchilla_fit_dolma
echo "Job $SLURM_JOB_ID  node=$SLURMD_NODENAME  started=$(date)"
PYTHONUNBUFFERED=1 /home/hamidieh/anaconda3/bin/python fit_triple_extrapolate.py
echo "finished=$(date)"
