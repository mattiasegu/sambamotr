#!/bin/bash

CONFIG=./configs/train_dancetrack.yaml
OUT_DIR=./outputs/memotr_dancetrack/
MODEL_NAME=memotr_dancetrack.pth
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
     --mode eval \
     --eval-mode specific \
     --data-root ${DATA_ROOT} \
     --eval-dir ${OUT_DIR} \
     --eval-model ${MODEL_NAME} \
     --eval-threads ${GPUS}
