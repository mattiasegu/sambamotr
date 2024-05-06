#!/bin/bash
set -x
TIME=24:00:00  # TIME=(24:00:00)
NNODES=2
GPUS_PER_NODE=4
CPUS_PER_TASK=4
MEM=200000
GPUS=`echo $GPUS_PER_NODE*$NNODES | bc`
SBATCH_ARGS=${SBATCH_ARGS:-""}

DATASET=dancetrack
JOB_NAME=${DATASET}/sambamotr_residual_masking_sync_longer
CONFIG=configs/sambamotr/${DATASET}/def_detr/train_residual_masking_sync_longer.yaml

# default scale for NUM_SAMPLES=8
LR=0.0002
LR_BACKBONE=0.00002
LR_POINTS=0.00001

JOB_NAME=${JOB_NAME}_lr_${LR}
OUT_DIR=/ptmp/msegu/workspaces/sambamotr/outputs/${JOB_NAME}/
CHECKPOINT=${OUT_DIR}last_checkpoint.pth

BS=1 
DATA_ROOT=/ptmp/msegu/data/

CMD=scripts/hpc/raven/tools/slurm_train.sh


GPUS_TYPE=a100
PORT=$(shuf -i 24000-29500 -n 1)
ID=$(sbatch \
     --parsable \
     -t ${TIME} \
     --job-name=${JOB_NAME} \
     --ntasks=${GPUS} \
     --ntasks-per-node=${GPUS_PER_NODE} \
     --gres=gpu:${GPUS_TYPE}:${GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     --mem=${MEM} \
     -e resources/errors/%j.log \
     -o resources/outputs/%j.log \
     ${SBATCH_ARGS} \
          ${CMD} \
          ${PORT} \
               --config-path ${CONFIG} \
               --outputs-dir ${OUT_DIR} \
               --batch-size ${BS} \
               --lr ${LR} \
               --lr-backbone ${LR_BACKBONE} \
               --lr-points ${LR_POINTS} \
               --data-root ${DATA_ROOT} \
               --pretrained-model pretrained/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \
               --launcher slurm \
               --use-distributed) 


for i in 1 2 3; do
PORT=$(shuf -i 24000-29500 -n 1)
ID=$(sbatch \
     --parsable \
     -t ${TIME} \
     --dependency=afterany:${ID}:+1 \
     --job-name=${JOB_NAME} \
     --ntasks=${GPUS} \
     --ntasks-per-node=${GPUS_PER_NODE} \
     --gres=gpu:${GPUS_TYPE}:${GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     --mem=${MEM} \
     -e resources/errors/%j.log \
     -o resources/outputs/%j.log \
     ${SBATCH_ARGS} \
          ${CMD} \
          ${PORT} \
               --config-path ${CONFIG} \
               --outputs-dir ${OUT_DIR} \
               --batch-size ${BS} \
               --lr ${LR} \
               --lr-backbone ${LR_BACKBONE} \
               --lr-points ${LR_POINTS} \
               --data-root ${DATA_ROOT} \
               --pretrained-model pretrained/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \
               --launcher slurm \
               --use-distributed \
               --resume ${CHECKPOINT} \
               --resume-scheduler True)
done