#!/bin/bash

JOB_NAME=debug_memotr_dancetrack
# CONFIG=./configs/sambamotr/train_dancetrack.yaml
# CONFIG=./configs/mambamotr/def_detr/train_dancetrack.yaml
# CONFIG=./configs/masked_mambamotr/def_detr/train_dancetrack_residual_masking_ref_pts.yaml
# CONFIG=./configs/masked_mambamotr/def_detr/train_dancetrack_residual_masking_dropout.yaml
# CONFIG=configs/masked_mambamotr/def_detr/train_dancetrack_residual_masking_sync_detach.yaml
# CONFIG=configs/masked_mambamotr/def_detr/train_dancetrack_residual_masking_sync.yaml
# CONFIG=configs/masked_mambamotr/def_detr/train_dancetrack_residual_masking_sync_longer.yaml
# CONFIG=configs/sambamotr/sportsmot/def_detr/train_masking_sync.yaml
# CONFIG=configs/sambamotr/dancetrack/def_detr/train_masking_sync.yaml
# CONFIG=configs/masked_mambamotr/def_detr/train_dancetrack_residual_masking_sync.yaml
CONFIG=configs/sambamotr/bft/def_detr/train_residual_masking_sync_longer.yaml
OUT_DIR=./outputs/tmp/${JOB_NAME}/
BS=1 
DATA_ROOT=/BS/diffusion-track/nobackup/data/mot

GPUS=1
# GPUS=2
if [ $GPUS -gt 1 ]
then
     CMD=tools/dist_main.sh
else
     CMD=tools/main.sh
fi

DATASET=BFT
SPLIT=val
EVAL_TRACKER=outputs/tmp/debug_memotr_dancetrack/val/checkpoint_0_tracker
# python \
# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client \
#      SparseTrackEval/scripts/run_bft.py --SPLIT_TO_EVAL val \
#      --METRICS HOTA CLEAR Identity  --GT_FOLDER ${DATA_ROOT}/${DATASET}/annotations_mot/${SPLIT} \
#      --GT_LOC_FORMAT {gt_folder}/{seq}.txt \
#      --SEQMAP_FILE ${DATA_ROOT}/${DATASET}/${SPLIT}_seqmap.txt \
#      --SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  \
#      --USE_PARALLEL False --NUM_PARALLEL_CORES 8 --PLOT_CURVES False \
#      --TRACKERS_FOLDER ${EVAL_TRACKER} --EVAL_INTERVAL 50 


# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client main.py \
#      --config-path ${CONFIG} \
#      --outputs-dir ${OUT_DIR} \
#      --batch-size ${BS} \
#      --data-root ${DATA_ROOT} \
#      --pretrained-model pretrained/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth

python main.py \
     --config-path ${CONFIG} \
     --outputs-dir ${OUT_DIR} \
     --batch-size ${BS} \
     --data-root ${DATA_ROOT} \
     --use-checkpoint \
     --pretrained-model pretrained/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth


# $CMD \
#      ${GPUS} \
#      --config-path ${CONFIG} \
#      --outputs-dir ${OUT_DIR} \
#      --batch-size ${BS} \
#      --data-root ${DATA_ROOT} \
#      --pretrained-model pretrained/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \
#      --use-checkpoint \
#      --launcher pytorch
     

# $CMD \
#      ${GPUS} \
#      --config-path ${CONFIG} \
#      --outputs-dir ${OUT_DIR} \
#      --batch-size ${BS} \
#      --data-root ${DATA_ROOT} \
#      --pretrained-model pretrained/samba/checkpoint_13.pth \
#      --use-checkpoint \
#      --launcher pytorch
     

# from dancetrack coco pretrained
# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client main.py \
#      --config-path ${CONFIG} \
#      --outputs-dir ${OUT_DIR} \
#      --batch-size ${BS} \
     # --pretrained-model pretrained/r50_deformable_detr_coco_dancetrack.pth \
#      --data-root ${DATA_ROOT} \
#      --launcher pytorch

# $CMD \
#      ${GPUS} \
#      --config-path ${CONFIG} \
#      --outputs-dir ${OUT_DIR} \
#      --batch-size ${BS} \
#      --data-root ${DATA_ROOT} \
#      --use-checkpoint \
#      --pretrained-model pretrained/r50_deformable_detr_coco_dancetrack.pth \
#      --launcher pytorch


# tools/main.sh \
#     ${GPUS} \
#      --config-path ${CONFIG} \
#      --outputs-dir ${OUT_DIR} \
#      --batch-size ${BS} \
#      --data-root ${DATA_ROOT}
#      # --use-checkpoint
