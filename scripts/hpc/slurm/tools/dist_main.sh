#!/usr/bin/env bash

set -x

PORT=$1
PY_ARGS=${@:2}

export MASTER_PORT=$PORT
echo "MASTER_PORT="$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "Node rank="$SLURM_NODEID
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE="$SLURM_GPUS_ON_NODE

# Make conda available
eval "$(conda shell.bash hook)"
# Activate a conda environment
conda activate sambamotr

export MPLBACKEND=Agg

# srun torchrun \
# srun -l --ntasks-per-node=1 torchrun \
srun -l --ntasks-per-node=1 python -m torch.distributed.run  \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main.py \
    --use-distributed \
    ${PY_ARGS}