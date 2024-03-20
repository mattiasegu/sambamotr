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

python main.py \
    ${@:2}