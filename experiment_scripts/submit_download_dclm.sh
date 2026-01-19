#!/bin/sh

#SBATCH -c 8 # Number of cores requested
#SBATCH -t 0-12:00 # Runtime in minutes
#SBATCH -p test # Partition to submit to
#SBATCH --mem=16G
#SBATCH -n 1
#SBATCH -o ../slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e ../slurm_out/slurm-%j.out # Standard err goes to this filehostname hostname
#SBATCH --account=dam_lab

# Change to the OLMo-core directory (parent of experiment_scripts)
cd /n/home05/sqin/OLMo-core

bash experiment_scripts/download_dclm_data.sh