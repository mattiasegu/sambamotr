#!/bin/bash
# JOB_NAME=eval_memotr_sportsmot
TIME=2:00:00  # TIME=(24:00:00)
GPUS=2
CPUS=16
MEM_PER_CPU=22000
SBATCH_ARGS=${SBATCH_ARGS:-""}
GPUS_PER_NODE=${GPUS}
CPUS_PER_TASK=${CPUS}

CONFIG=./configs/sambamotr/mot17/def_detr/train_residual_masking_sync_longer.yaml
OUT_DIR=./pretrained/sambamotr/mot17/sambamotr_residual_masking_sync_longer_sched2_lr_0.0002
MODEL_NAME=checkpoint_74.pth
BS=1 
DATA_ROOT=/BS/diffusion-track/nobackup/data/mot/

### FULL EVAL
EVAL_INTERVAL=1
# DET_SCORE_THRESH=0.7
# TRACK_SCORE_THRESH=0.6
# MISS_TOLERANCE=15
# EXP_NAME=interval_${EVAL_INTERVAL}

CMD=tools/main.sh


PARTITION=gpu22  # PARTITION=(gpu16 | gpu20 | gpu22) 
for DET_SCORE_THRESH in 0.5; do
for TRACK_SCORE_THRESH in 0.5; do
for MISS_TOLERANCE in 15; do
     EXP_NAME=det_${DET_SCORE_THRESH}_track_${TRACK_SCORE_THRESH}_miss_${MISS_TOLERANCE}_interval_${EVAL_INTERVAL}
     ID=$(sbatch \
          --parsable \
          -t ${TIME} \
          --job-name=${JOB_NAME} \
          -p ${PARTITION} \
          --gres=gpu:${GPUS_PER_NODE} \
          -e resources/errors/%j.log \
          -o resources/outputs/%j.log \
          ${SBATCH_ARGS} \
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
               --eval-threads ${GPUS} \
               --eval-data-split test \
               --det-score-thresh ${DET_SCORE_THRESH} \
               --track-score-thresh ${TRACK_SCORE_THRESH} \
               --result-score-thresh ${TRACK_SCORE_THRESH} \
               --update-thresh ${TRACK_SCORE_THRESH} \
               --miss-tolerance ${MISS_TOLERANCE})
done
done
done

# for DET_SCORE_THRESH in 0.5; do
# for TRACK_SCORE_THRESH in 0.5; do
# for MISS_TOLERANCE in 15; do
#      EXP_NAME=det_${DET_SCORE_THRESH}_track_${TRACK_SCORE_THRESH}_miss_${MISS_TOLERANCE}_interval_${EVAL_INTERVAL}
#      python -m debugpy --listen $HOSTNAME:5678 --wait-for-client main.py \
#           --mode eval \
#           --eval-mode specific \
#           --config-path ${CONFIG} \
#           --data-root ${DATA_ROOT} \
#           --eval-dir ${OUT_DIR} \
#           --eval-model ${MODEL_NAME} \
#           --exp-name ${EXP_NAME} \
#           --eval-interval ${EVAL_INTERVAL} \
#           --eval-threads ${GPUS} \
#           --eval-data-split train \
#           --det-score-thresh ${DET_SCORE_THRESH} \
#           --track-score-thresh ${TRACK_SCORE_THRESH} \
#           --result-score-thresh ${TRACK_SCORE_THRESH} \
#           --update-thresh ${TRACK_SCORE_THRESH} \
#           --miss-tolerance ${MISS_TOLERANCE}


# done
# done
# done

