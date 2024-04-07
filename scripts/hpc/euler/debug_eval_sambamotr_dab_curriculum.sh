#!/bin/bash
JOB_NAME=sambamotr_curriculum_dancetrack
TIME=1:00:00  # TIME=(24:00:00)
GPUS=8
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

CONFIG=./configs/sambamotr_curriculum/train_dancetrack.yaml
OUT_DIR=/cluster/work/cvl/segum/workspaces/sambamotr/outputs/${JOB_NAME}/
BS=1 
DATA_ROOT=/cluster/work/cvl/segum/datasets/mot/data/
MODEL_NAME=checkpoint_19.pth

CMD=scripts/hpc/euler/tools/main.sh

echo "Launching ${CMD} on ${GPUS} gpus."
echo "Starting job ${JOB_NAME} from ${CONFIG}" 

mkdir -p resources/errors/ 
mkdir -p resources/outputs/


# ${CMD} \
# ${GPUS} \
#      --mode eval \
#      --eval-mode specific \
#      --config-path ${CONFIG} \
#      --data-root ${DATA_ROOT} \
#      --eval-dir ${OUT_DIR} \
#      --eval-model ${MODEL_NAME} \
#      --eval-threads ${GPUS}

${CMD} \
${GPUS} \
     --mode eval \
     --eval-mode continue \
     --config-path ${CONFIG} \
     --data-root ${DATA_ROOT} \
     --eval-dir ${OUT_DIR} \
     --eval-threads ${GPUS}
