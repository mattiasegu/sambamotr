#!/bin/bash

JOB_NAME=debug_memotr_dancetrack
CONFIG=./configs/sambamotr/train_dancetrack.yaml
OUT_DIR=./outputs/tmp/${JOB_NAME}/
BS=1 
DATA_ROOT=/BS/diffusion-track/nobackup/data/mot/

GPUS=1
if [ $GPUS -gt 1 ]
then
     CMD=tools/dist_main.sh
else
     CMD=tools/main.sh
fi


python -m debugpy --listen $HOSTNAME:5678 --wait-for-client main.py \
     --config-path ${CONFIG} \
     --outputs-dir ${OUT_DIR} \
     --batch-size ${BS} \
     --data-root ${DATA_ROOT}
     # --use-checkpoint


# tools/main.sh \
#     ${GPUS} \
#      --config-path ${CONFIG} \
#      --outputs-dir ${OUT_DIR} \
#      --batch-size ${BS} \
#      --data-root ${DATA_ROOT}
#      # --use-checkpoint