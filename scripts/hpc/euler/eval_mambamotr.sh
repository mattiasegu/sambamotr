#!/bin/bash
JOB_NAME=eval_mambamotr_dancetrack
TIME=1:00:00  # TIME=(24:00:00)
GPUS=2
CPUS=16
MEM_PER_CPU=10000
SBATCH_ARGS=${SBATCH_ARGS:-""}
GPUS_PER_NODE=${GPUS}
CPUS_PER_TASK=${CPUS}

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

CONFIG=./configs/masked_mambamotr/def_detr/train_dancetrack_residual_masking.yaml
OUT_DIR=/cluster/work/cvl/segum/workspaces/sambamotr/outputs/${JOB_NAME}/
BS=1 
DATA_ROOT=/cluster/work/cvl/segum/datasets/mot/data/
MODEL_NAME=checkpoint_15.pth

CMD=scripts/hpc/euler/tools/main.sh

echo "Launching ${CMD} on ${GPUS} gpus."
echo "Starting job ${JOB_NAME} from ${CONFIG}" 

mkdir -p resources/errors/ 
mkdir -p resources/outputs/


# GPUS_TYPE=rtx_4090  # GPUS_TYPE=(rtx_3090 | rtx_4090 | titan_rtx)
# ID=$(sbatch \
#      --parsable \
#      -t ${TIME} \
#      --job-name=${JOB_NAME} \
#      --gpus=${GPUS_TYPE}:${GPUS_PER_NODE} \
#      --ntasks=${CPUS_PER_TASK} \
#      --mem-per-cpu ${MEM_PER_CPU} \
#      -e resources/errors/%j.log \
#      -o resources/outputs/%j.log \
#      ${SBATCH_ARGS} \
#      ${CMD} \
#      ${GPUS} \
#           --mode eval \
#           --eval-mode specific \
#           --config-path ${CONFIG} \
#           --data-root ${DATA_ROOT} \
#           --eval-dir ${OUT_DIR} \
#           --eval-model ${MODEL_NAME} \
#           --eval-threads ${GPUS})

GPUS_TYPE=rtx_4090  # GPUS_TYPE=(rtx_3090 | rtx_4090 | titan_rtx)
ID=$(sbatch \
     --parsable \
     -t ${TIME} \
     --job-name=${JOB_NAME} \
     --gpus=${GPUS_TYPE}:${GPUS_PER_NODE} \
     --ntasks=${CPUS_PER_TASK} \
     --mem-per-cpu ${MEM_PER_CPU} \
     -e resources/errors/%j.log \
     -o resources/outputs/%j.log \
     ${SBATCH_ARGS} \
     ${CMD} \
     ${GPUS} \
          --mode eval \
          --eval-mode continue \
          --config-path ${CONFIG} \
          --data-root ${DATA_ROOT} \
          --eval-dir ${OUT_DIR} \
          --eval-threads ${GPUS})
