#!/bin/bash

CONFIG=./configs/masked_mambamotr/def_detr/train_dancetrack_residual_masking_only_pos_fps.yaml
OUT_DIR=./pretrained/sambamotr_residual_masking_only_pos_fps
# CONFIG=./configs/masked_mambamotr/def_detr/train_dancetrack_residual_masking_only_pos.yaml
# OUT_DIR=./pretrained/sambamotr_residual_masking_only_pos
MODEL_NAME=checkpoint_12.pth
EVAL_INTERVAL=10
# EVAL_INTERVAL=100
EXP_NAME=interval_${EVAL_INTERVAL}
BS=1 
DATA_ROOT=/BS/diffusion-track/nobackup/data/mot/

GPUS=1
if [ $GPUS -gt 1 ]
then
     CMD=tools/dist_main.sh
else
     CMD=tools/main.sh
fi

### FULL EVAL

${CMD} \
    ${GPUS} \
     --mode eval \
     --eval-mode specific \
     --config-path ${CONFIG} \
     --data-root ${DATA_ROOT} \
     --eval-dir ${OUT_DIR} \
     --eval-model ${MODEL_NAME} \
     --exp-name ${EXP_NAME} \
     --eval-interval ${EVAL_INTERVAL} \
     --eval-threads ${GPUS}


# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client main.py \
#      --mode eval \
#      --eval-mode specific \
#      --config-path ${CONFIG} \
#      --data-root ${DATA_ROOT} \
#      --eval-dir ${OUT_DIR} \
#      --eval-model ${MODEL_NAME} \
#      --exp-name ${EXP_NAME} \
#      --eval-interval ${EVAL_INTERVAL} \
#      --eval-threads ${GPUS}

### SUBMIT ONLY

# ${CMD} \
#     ${GPUS} \
#      --mode submit \
#      --data-root ${DATA_ROOT} \
#      --submit-dir ${OUT_DIR} \
#      --submit-model ${MODEL_NAME} \
#      --config-path ${CONFIG} \
#      --exp-name ${EXP_NAME} \
#      --eval-interval ${EVAL_INTERVAL} 

# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client main.py \
#      --mode submit \
#      --data-root ${DATA_ROOT} \
#      --submit-dir ${OUT_DIR} \
#      --submit-data-split val \
#      --submit-model ${MODEL_NAME} \
#      --config-path ${CONFIG} \
#      --exp-name ${EXP_NAME} \
#      --eval-interval ${EVAL_INTERVAL} 

### TRACKEVAL ONLY
DATA_ROOT=/BS/diffusion-track/nobackup/data/mot/
SPLIT=val
DATASET=DanceTrack
EVAL_TRACKER=pretrained/sambamotr_residual_masking_only_pos_fps/val/interval_100/tracker
# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client \
#      SparseTrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL val \
#      --METRICS HOTA CLEAR Identity  --GT_FOLDER ${DATA_ROOT}/${DATASET}/${SPLIT} \
#      --SEQMAP_FILE ${DATA_ROOT}/${DATASET}/${SPLIT}_seqmap.txt \
#      --SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  \
#      --USE_PARALLEL True --NUM_PARALLEL_CORES 8 --PLOT_CURVES False \
#      --TRACKERS_FOLDER ${EVAL_TRACKER} --EVAL_INTERVAL 100
# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client \
# python \
#      SparseTrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL val \
#      --METRICS HOTA CLEAR Identity  --GT_FOLDER ${DATA_ROOT}/${DATASET}/${SPLIT} \
#      --SEQMAP_FILE ${DATA_ROOT}/${DATASET}/${SPLIT}_seqmap.txt \
#      --SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  \
#      --USE_PARALLEL False --NUM_PARALLEL_CORES 8 --PLOT_CURVES False \
#      --TRACKERS_FOLDER ${EVAL_TRACKER} --EVAL_INTERVAL 100
