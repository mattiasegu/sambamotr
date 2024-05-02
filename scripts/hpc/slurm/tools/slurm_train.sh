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

srun python -u main.py \
    --use-distributed \
    ${PY_ARGS}
    