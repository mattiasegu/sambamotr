#!/bin/bash

CONFIG=./configs/sambamotr/train_dancetrack.yaml
OUT_DIR=./outputs/tmp/sambamotr_dancetrack/
MODEL_NAME=checkpoint_0.pth
BS=1 
DATA_ROOT=/BS/diffusion-track/nobackup/data/mot/
SPLIT=val
DATASET=DanceTrack
EVAL_TRACKER=outputs/tmp/debug_memotr_dancetrack/val/checkpoint_0_tracker

GPUS=1
if [ $GPUS -gt 1 ]
then
     CMD=tools/dist_main.sh
else
     CMD=tools/main.sh
fi


# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client \
python \
     TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL ${SPLIT} \
     --METRICS HOTA CLEAR Identity  --GT_FOLDER ${DATA_ROOT}/${DATASET}/${SPLIT} \
     --SEQMAP_FILE ${DATA_ROOT}/${DATASET}/${SPLIT}_seqmap.txt \
     --SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  \
     --USE_PARALLEL True --NUM_PARALLEL_CORES 8 --PLOT_CURVES False \
     --TRACKERS_FOLDER ${EVAL_TRACKER}




