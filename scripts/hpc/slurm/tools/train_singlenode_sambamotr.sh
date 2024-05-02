#!/bin/bash
JOB_NAME=multinode_sambamotr_dancetrack_def_detr
TIME=1:00:00  # TIME=(24:00:00)
PARTITION=gpu22  # PARTITION=(gpu16 | gpu20 | gpu22) 

NNODES=1
GPUS_PER_NODE=1
CPUS_PER_TASK=2
MEM_PER_CPU=10000
GPUS=`echo $GPUS_PER_NODE*$NNODES | bc`
SBATCH_ARGS=${SBATCH_ARGS:-""}


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

CONFIG=configs/sambamotr/dancetrack/def_detr/train_masking_sync.yaml
OUT_DIR=./outputs/${JOB_NAME}/
BS=1 
DATA_ROOT=/BS/diffusion-track/nobackup/data/mot/

CMD=scripts/hpc/slurm/tools/slurm_train.sh

echo "Launching ${CMD} on ${GPUS} gpus."
echo "Starting job ${JOB_NAME} from ${CONFIG}" 

mkdir -p resources/errors/ 
mkdir -p resources/outputs/ 

# GPUS_TYPE=a40
PORT=$(shuf -i 24000-29500 -n 1)
     # --gres=gpu:${GPUS_TYPE}:${GPUS_PER_NODE} \
ID=$(sbatch \
     --parsable \
     -t ${TIME} \
     --job-name=${JOB_NAME} \
     -p ${PARTITION} \
     --ntasks=${GPUS} \
     --ntasks-per-node=${GPUS_PER_NODE} \
     --gres=gpu:${GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     --mem-per-cpu=${MEM_PER_CPU} \
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
          --use-checkpoint \
          --pretrained-model pretrained/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \
          --use-distributed \
          --launcher slurm)