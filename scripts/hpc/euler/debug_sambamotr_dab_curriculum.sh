#!/bin/bash
JOB_NAME=debug_sambamotr_curriculum_dancetrack
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

CONFIG=./configs/sambamotr_curriculum/train_dancetrack.yaml
OUT_DIR=/cluster/work/cvl/segum/workspaces/sambamotr/outputs/tmp/${JOB_NAME}/
BS=1 
DATA_ROOT=/cluster/work/cvl/segum/datasets/mot/data/

if [ $GPUS -gt 1 ]
then
     CMD=scripts/hpc/euler/tools/dist_main.sh
else
     CMD=scripts/hpc/euler/tools/main.sh
fi

echo "Launching ${CMD} on ${GPUS} gpus."
echo "Starting job ${JOB_NAME} from ${CONFIG}" 

mkdir -p resources/errors/ 
mkdir -p resources/outputs/

${CMD} \
${GPUS} \
     --config-path ${CONFIG} \
     --outputs-dir ${OUT_DIR} \
     --batch-size ${BS} \
     --lr ${LR} \
     --lr-backbone ${LR_BACKBONE} \
     --lr-points ${LR_POINTS} \
     --data-root ${DATA_ROOT} \
     --use-checkpoint