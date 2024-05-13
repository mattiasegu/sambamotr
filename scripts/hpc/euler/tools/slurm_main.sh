#!/usr/bin/env bash
module load gcc/8.2.0 python/3.11.2 cuda/11.8.0 nccl/2.16.2-1 cudnn/8.4.0.27

# load venv
WORKSPACE=/cluster/work/cvl/segum/workspaces
export PYTHONPATH=${WORKSPACE}/sambamotr/venv/sambamotr/bin/python
source ${WORKSPACE}/sambamotr/venv/sambamotr/bin/activate

PORT=$1
PYARGS=${@:2}

export MASTER_PORT=$PORT
echo "MASTER_PORT="$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "Node rank="$SLURM_NODEID
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE="$SLURM_GPUS_ON_NODE


export MPLBACKEND=Agg

echo ${PYARGS}

srun -c 4 --kill-on-bad-exit=1 python -u main.py ${PYARGS}
    