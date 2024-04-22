#!/bin/bash

JOB_NAME=debug_memotr_dancetrack
# CONFIG=./configs/sambamotr/train_dancetrack.yaml
# CONFIG=./configs/mambamotr/def_detr/train_dancetrack.yaml
# CONFIG=./configs/masked_mambamotr/def_detr/train_dancetrack_residual_masking_ref_pts.yaml
CONFIG=./configs/masked_mambamotr/def_detr/train_dancetrack_residual_masking_dropout.yaml
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
     --data-root ${DATA_ROOT} \
     --pretrained-model pretrained/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth

# python main.py \
#      --config-path ${CONFIG} \
#      --outputs-dir ${OUT_DIR} \
#      --batch-size ${BS} \
#      --data-root ${DATA_ROOT} \
#      --use-checkpoint \
#      --pretrained-model pretrained/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth


# tools/main.sh \
#     ${GPUS} \
#      --config-path ${CONFIG} \
#      --outputs-dir ${OUT_DIR} \
#      --batch-size ${BS} \
#      --data-root ${DATA_ROOT}
#      # --use-checkpoint
