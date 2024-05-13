#!/bin/bash
# TIME=120:00:00  # TIME=(24:00:00)
# NNODES=1
# GPUS_PER_NODE=8
TIME=120:00:00  # TIME=(24:00:00)
NNODES=1
GPUS_PER_NODE=8
CPUS_PER_TASK=4
MEM_PER_CPU=10000
GPUS=`echo $GPUS_PER_NODE*$NNODES | bc`
SBATCH_ARGS=${SBATCH_ARGS:-""}

DATASET=bft
JOB_NAME=sambamotr_residual_masking_sync_longer_dropout
CONFIG=./configs/sambamotr/${DATASET}/def_detr/train_residual_masking_sync_longer_dropout.yaml

# rescale
BS_PER_GPU=1
BS=`echo $GPUS*$BS_PER_GPU | bc`
#########################
# BASE LR PARAMETERS
# LR=0.0002
# LR_BACKBONE=0.00002
# LR_POINTS=0.00001
# LR_PER_SAMPLE=0.0002/8
# NUM_SAMPLES=BS=8
#########################

# default scale for NUM_SAMPLES=8
LR=0.0002
LR_BACKBONE=0.00002
LR_POINTS=0.00001
JOB_NAME=${JOB_NAME}_lr_${LR}

OUT_DIR=/cluster/work/cvl/segum/workspaces/sambamotr/outputs/${DATASET}/${JOB_NAME}/
BS=1 
DATA_ROOT=/cluster/work/cvl/segum/datasets/mot/data/

CMD=scripts/hpc/euler/sambamotr/bft/bft_main.sh

echo "Launching ${CMD} on ${GPUS} gpus."
echo "Starting job ${JOB_NAME} from ${CONFIG}" 

mkdir -p resources/errors/ 
mkdir -p resources/outputs/

     # --gpus=${GPUS_TYPE}:${GPUS_PER_NODE} \
     # --gpus=${GPUS_PER_NODE} \
     # --gres=gpumem:20g \

GPUS_TYPE=rtx_4090  # GPUS_TYPE=(rtx_3090 | rtx_4090 | titan_rtx)
PORT=$(shuf -i 24000-29500 -n 1)
ID=$(sbatch \
     --parsable \
     -t ${TIME} \
     --job-name=${JOB_NAME} \
     --ntasks=${GPUS} \
     --ntasks-per-node=${GPUS_PER_NODE} \
     --gpus=${GPUS_TYPE}:${GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     --mem-per-cpu ${MEM_PER_CPU} \
     -e resources/errors/%j.log \
     -o resources/outputs/%j.log \
     ${SBATCH_ARGS} \
     ${CMD} \
          ${PORT} \
          ${GPUS} \
          --config-path ${CONFIG} \
          --outputs-dir ${OUT_DIR} \
          --batch-size ${BS} \
          --lr ${LR} \
          --lr-backbone ${LR_BACKBONE} \
          --lr-points ${LR_POINTS} \
          --data-root ${DATA_ROOT} \
          --use-checkpoint \
          --pretrained-model pretrained/deformable_detr.pth \
          --launcher pytorch \
          --use-distributed)
