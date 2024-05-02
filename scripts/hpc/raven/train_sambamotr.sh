#!/bin/bash
DATASET=dancetrack
JOB_NAME=${DATASET}/debug_sambamotr_masking_sync
CONFIG=configs/sambamotr/${DATASET}/def_detr/train_masking_sync.yaml
OUT_DIR=/ptmp/msegu/workspaces/sambamotr/outputs/tmp/${JOB_NAME}/
BS=1 
DATA_ROOT=/ptmp/msegu/data/

GPUS=1
# GPUS=2
CMD=scripts/hpc/raven/tools/slurm_train.sh

${CMD} \
${PORT} \
     --config-path ${CONFIG} \
     --outputs-dir ${OUT_DIR} \
     --batch-size ${BS} \
     --lr ${LR} \
     --lr-backbone ${LR_BACKBONE} \
     --lr-points ${LR_POINTS} \
     --data-root ${DATA_ROOT} \
     --use-checkpoint \
     --pretrained-model pretrained/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \
     --launcher slurm \
     # --use-distributed 
