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

# load modules
module load cuda/12.1 cudnn/8.8.1 nccl/2.17.1

# load venv
WORKSPACE=/ptmp/msegu/workspaces/sambamotr
export PYTHONPATH=${WORKSPACE}/venv/sambamotr/bin/python
source ${WORKSPACE}/venv/sambamotr/bin/activate

export MPLBACKEND=Agg

python -u main.py \
    ${PY_ARGS}
    