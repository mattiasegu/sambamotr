#!/usr/bin/env bash

GPUS=$1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PORT=$(shuf -i 24000-29500 -n 1)

# Make conda available
eval "$(conda shell.bash hook)"
# Activate a conda environment
conda activate sambamotr

export MPLBACKEND=Agg

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    main.py \
    --use-distributed \
    ${@:2}