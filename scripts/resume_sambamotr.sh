#!/bin/bash

CONFIG=./configs/sambamotr/train_dancetrack.yaml
OUT_DIR=./outputs/tmp/sambamotr_dancetrack/
BS=1 
DATA_ROOT=/BS/diffusion-track/nobackup/data/mot/
CHECKPOINT=outputs/tmp/sambamotr_dancetrack/checkpoint_0.pth

GPUS=1
if [ $GPUS -gt 1 ]
then
     CMD=tools/dist_main.sh
else
     CMD=tools/main.sh
fi

# ${CMD} \
#     ${GPUS} \
#      --config-path ${CONFIG} \
#      --outputs-dir ${OUT_DIR} \
#      --batch-size ${BS} \
#      --data-root ${DATA_ROOT} \
#      --use-checkpoint \
#      --resume ${CHECKPOINT} \ 
#      --resume-scheduler True

python -m debugpy --listen $HOSTNAME:5678 --wait-for-client main.py \
     --config-path ${CONFIG} \
     --outputs-dir ${OUT_DIR} \
     --batch-size ${BS} \
     --data-root ${DATA_ROOT} \
     --use-checkpoint \
     --resume ${CHECKPOINT} \
     --resume-scheduler True
