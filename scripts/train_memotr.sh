#!/bin/bash

CONFIG=./configs/train_dancetrack.yaml
OUT_DIR=./outputs/tmp/memotr_dancetrack/
BS=1 
DATA_ROOT=/BS/diffusion-track/nobackup/data/mot/

GPUS=1
if [ $GPUS -gt 1 ]
then
     CMD=tools/dist_main.sh
else
     CMD=tools/main.sh
fi

${CMD} \
    ${GPUS} \
     --config-path ${CONFIG} \
     --outputs-dir ${OUT_DIR} \
     --batch-size ${BS} \
     --data-root ${DATA_ROOT} \
     --use-checkpoint
