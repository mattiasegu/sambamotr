#!/bin/bash
DATASET=dancetrack
JOB_NAME=sambamotr_masking_sync
CONFIG=./configs/sambamotr/${DATASET}/def_detr/train_masking_sync.yaml
WORKSPACE=/cluster/work/cvl/lpiccinelli/sambamotr/

TIME=24:00:00  # TIME=(24:00:00)
MEM_PER_CPU=4000
SBATCH_ARGS=${SBATCH_ARGS:-""}
GPUS=8
GPUS_PER_NODE=4
CPUS_PER_TASK=4

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

# CHANGE WORKSPACE!
OUT_DIR=${WORKSPACE}/outputs/${DATASET}/${JOB_NAME}/
DATA_ROOT=/cluster/project/cvl/lpiccinelli/sambamotr/data
BS=1 

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


GPUS_TYPE=rtx_3090  # GPUS_TYPE=(rtx_3090 | rtx_4090 | titan_rtx | quadro_rtx_6000)
ID=$(sbatch \
     --parsable \
     -t ${TIME} \
     --job-name=${JOB_NAME} \
     --gpus-per-node=${GPUS_TYPE}:${GPUS_PER_NODE} \
     --ntasks=${GPUS} \
     --ntasks-per-node=${GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     --mem-per-cpu=${MEM_PER_CPU} \
     -e resources/errors/%j.log \
     -o resources/outputs/%j.log \
     ${SBATCH_ARGS} \
     ${CMD} \
     --config-path ${CONFIG} \
     --outputs-dir ${OUT_DIR} \
     --batch-size ${BS} \
     --lr ${LR} \
     --lr-backbone ${LR_BACKBONE} \
     --lr-points ${LR_POINTS} \
     --data-root ${DATA_ROOT} \
     --pretrained-model /cluster/project/cvl/lpiccinelli/sambamotr/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth)