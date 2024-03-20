#!/bin/bash
JOB_NAME=memotr_dancetrack
TIME=1:00:00  # TIME=(24:00:00)
PARTITION=gpu22  # PARTITION=(gpu16 | gpu20 | gpu22) 
GPUS=2
CPUS=16
MEM_PER_CPU=22000
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

# rescale for NUM_SAMPLES=4
LR=0.0001
LR_BACKBONE=0.00001
LR_POINTS=0.000005
JOB_NAME=${JOB_NAME}_lr_${LR}

CONFIG=./configs/train_dancetrack.yaml
OUT_DIR=./outputs/${JOB_NAME}/
BS=1 
DATA_ROOT=/BS/diffusion-track/nobackup/data/mot/

if [ $GPUS -gt 1 ]
then
     CMD=tools/dist_main.sh
else
     CMD=tools/main.sh
fi

echo "Launching ${CMD} on ${GPUS} gpus."
echo "Starting job ${JOB_NAME} from ${CONFIG}" 

mkdir -p resources/errors/ 
mkdir -p resources/outputs/ 


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
          --config-path ${CONFIG} \
          --outputs-dir ${OUT_DIR} \
          --batch-size ${BS} \
          --lr ${LR} \
          --lr-backbone ${LR_BACKBONE} \
          --lr-points ${LR_POINTS} \
          --data-root ${DATA_ROOT} \
          --use-checkpoint)