#!/bin/bash

#SBATCH -p kempner_h100 # Partition to submit to
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks across all nodes
#SBATCH --cpus-per-task=24        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=250G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --account=kempner_dam_lab
#SBATCH -o slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e slurm_out/slurm-%j.out # Standard err goes to this file
#SBATCH --job-name=olmo-multi    # create a short name for your job


module purge
module load Mambaforge
module load cuda cudnn
mamba activate openrlhf


# Print job info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_PROCID: $SLURM_PROCID"
echo ""

cd ../

export CACHED_PATH_CACHE_ROOT=/n/netscratch/dam_lab/Lab/sqin/olmo/

# Get the master node address (same approach as Ray script)
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=( $nodes )
MASTER_NODE=${nodes_array[0]}
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname --ip-address)
MASTER_PORT=12345

echo "Master node: $MASTER_NODE"
echo "Master IP: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo ""

# Debug: Check node information
echo "=== Debugging node information ==="
srun bash -c 'echo "Node: $(hostname), SLURM_PROCID: $SLURM_PROCID, SLURM_NODEID: $SLURM_NODEID, SLURM_LOCALID: $SLURM_LOCALID"'
echo ""

echo "=== Network interfaces available ==="
srun bash -c 'echo "Node: $(hostname)"; ip addr show | grep -E "^[0-9]+: " | awk "{print \$2}" | sed "s/:$//"'
echo ""

echo "=== Network IPs (excluding loopback) ==="
srun bash -c 'echo "Node: $(hostname)"; ip addr show | grep "inet " | grep -v "127.0.0.1" | awk "{print \$NF, \$2}"'
echo ""

# Enable distributed debugging (set to INFO only when debugging issues)
# TORCH_DISTRIBUTED_DEBUG=INFO  # Uncomment for verbose FSDP/DDP debugging
# export NCCL_DEBUG=INFO         # Uncomment for verbose NCCL debugging
export NCCL_DEBUG=WARN           # Only show NCCL warnings/errors

# Network configuration for InfiniBand
export NCCL_SOCKET_IFNAME=^docker0,lo  # Exclude loopback and docker
export NCCL_IB_DISABLE=0               # Enable InfiniBand (0 = enabled)
export NCCL_IB_HCA=mlx5                # InfiniBand adapter (usually mlx5 for modern cards)
export NCCL_IB_GID_INDEX=3             # RoCE mode (3 is common for RoCEv2)
# If you want to explicitly use InfiniBand interfaces, uncomment:
# export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3

echo "=== Launching distributed training ==="
# Run distributed training
# Note: Using SLURM_PROCID instead of SLURM_NODEID for node_rank
srun torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /n/home05/sqin/OLMo/scripts/train.py /n/home05/sqin/OLMo/experiment_scripts/OLMo2-7B-stage2-custom.yaml
