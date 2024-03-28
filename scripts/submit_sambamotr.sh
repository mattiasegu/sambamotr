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

# ${CMD} \
#     ${GPUS} \
     # --mode submit \
     # --data-root ${DATA_ROOT} \
     # --submit-dir ${OUT_DIR} \
     # --submit-model ${MODEL_NAME} \
     # --config-path ${CONFIG}

python -m debugpy --listen $HOSTNAME:5678 --wait-for-client main.py \
     --mode submit \
     --data-root ${DATA_ROOT} \
     --submit-dir ${OUT_DIR} \
     --submit-model ${MODEL_NAME} \
     --config-path ${CONFIG}