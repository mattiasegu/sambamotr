#!/usr/bin/env bash
module load gcc/8.2.0 python/3.11.2 cuda/11.8.0 nccl/2.16.2-1 cudnn/8.4.0.27

# load venv
WORKSPACE=/cluster/work/cvl/segum/workspaces
export PYTHONPATH=${WORKSPACE}/sambamotr/venv/sambamotr/bin/python
source ${WORKSPACE}/sambamotr/venv/sambamotr/bin/activate


PORT=$1
GPUS=$2
PYARGS=${@:3}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

export MASTER_PORT=$PORT
echo "MASTER_PORT="$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "Node rank="$SLURM_NODEID
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE="$SLURM_GPUS_ON_NODE


##############################
##### Data preprocessing
##############################
bash scripts/hpc/euler/sambamotr/mot17/mot17_copy.sh

DATA_ROOT=${TMPDIR}/data

##############################
##### Argument manipulation (override DATA_ROOT)
##############################

# Initialize an empty string for the new arguments
NEW_PYARGS=""

# Flag to skip the next argument
SKIP_NEXT=0

# Loop through the PYARGS
for arg in $PYARGS; do
    # If the next argument should be skipped, just toggle the flag and continue
    if [ $SKIP_NEXT -eq 1 ]; then
        SKIP_NEXT=0
        continue
    fi
    
    # Check if the argument is --data-root
    if [ "$arg" == "--data-root" ]; then
        # Add modified --data-root with new value to NEW_PYARGS
        NEW_PYARGS+="--data-root $DATA_ROOT "
        # Toggle the flag to skip the next argument since we're replacing it
        SKIP_NEXT=1
    else
        # If not --data-root, just add the argument to NEW_PYARGS
        NEW_PYARGS+="$arg "
    fi
done
echo $NEW_PYARGS
###############

export MPLBACKEND=Agg

echo ${PYARGS}

# srun -c 4 --kill-on-bad-exit=1 python -u main.py ${NEW_PYARGS}
  
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    main.py \
    ${NEW_PYARGS}
