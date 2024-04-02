#!/bin/bash

CONFIG=./configs/sambamotr/train_dancetrack.yaml
OUT_DIR=./outputs/tmp/sambamotr_dancetrack/
MODEL_NAME=checkpoint_0.pth
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
     --config-path ${CONFIG} \
     --data-root ${DATA_ROOT} \
     --eval-dir ${OUT_DIR} \
     --eval-model ${MODEL_NAME} \
     --eval-threads ${GPUS}


# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client main.py \
#      --mode eval \
#      --eval-mode specific \
#      --config-path ${CONFIG} \
#      --data-root ${DATA_ROOT} \
#      --eval-dir ${OUT_DIR} \
#      --eval-model ${MODEL_NAME} \
#      --eval-threads ${GPUS}



