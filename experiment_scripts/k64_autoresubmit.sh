#!/bin/bash
#SBATCH --job-name=k64-watcher
#SBATCH --partition=intermediate
#SBATCH --account=barak_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00
#SBATCH -o ../slurm_out/k64-watcher-%j.out
#SBATCH -e ../slurm_out/k64-watcher-%j.out

# Auto-resubmit watcher for the big 2-GPU K=64 rungs.
# Every 30 min: if a rung is not in the queue and its last job state is not
# COMPLETED, resubmit it (2-GPU) so it resumes from checkpoint. Stops when all done.
set -u
cd /n/home05/sqin/OLMo-core/experiment_scripts
RUNGS=( "190M 1" "190M 2" "370M 0.5" "370M 1" )
declare -A TRIES
MAXTRIES=8   # safety cap per rung (crash loop guard)

submit() {  # size chin
  sbatch --parsable --job-name="${2}-64K-${1}" \
    --partition=gpu_h200 --account=barak_lab --time=3-00:00 --exclude=holygpu8a12501 \
    --gres=gpu:2 --cpus-per-task=32 \
    --export=ALL,MODEL_SIZE=${1},CHIN=${2},NUM_SEEDS=64,MICROBATCH_MULT=32,NGPU=2,LR=3e-3,WD=0.1 \
    train_scale_paraphrase.sh
}

while true; do
  alldone=1
  for j in "${RUNGS[@]}"; do
    read -r size chin <<< "$j"
    name="${chin}-64K-${size}"
    # in queue (running or pending)?
    if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "$name"; then alldone=0; continue; fi
    # most recent job state for this name
    st=$(sacct -u "$USER" -S 2026-07-18 -o JobName%20,State%20,End -n 2>/dev/null \
         | awk -v n="$name" '$1==n{s=$2} END{print s}')
    if [ "$st" = "COMPLETED" ]; then continue; fi   # this rung is finished
    # needs (re)submitting
    t=${TRIES[$name]:-0}
    if [ "$t" -ge "$MAXTRIES" ]; then
      echo "[$(date +%F_%T)] $name hit MAXTRIES ($MAXTRIES), last state=$st — NOT resubmitting"; continue
    fi
    id=$(submit "$size" "$chin")
    TRIES[$name]=$((t+1))
    alldone=0
    echo "[$(date +%F_%T)] resubmitted $name (try $((t+1)), prev state=${st:-none}) -> job $id"
  done
  if [ "$alldone" = "1" ]; then echo "[$(date +%F_%T)] ALL 4 rungs COMPLETED — watcher exiting"; break; fi
  sleep 1800
done
