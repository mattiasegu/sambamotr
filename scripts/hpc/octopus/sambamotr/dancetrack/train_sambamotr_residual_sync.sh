#!/bin/bash
DATASET=dancetrack
JOB_NAME=sambamotr_residual
CONFIG=./configs/sambamotr/${DATASET}/def_detr/train_residual_sync.yaml

# rescale
BS_PER_GPU=1
BS=`echo $GPUS*$BS_PER_GPU | bc`
GPUS=8

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

OUT_DIR=/srv/beegfs02/scratch/3d_tracking/data/video_depth/samba/outputs/${DATASET}/${JOB_NAME}/
BS=1 
DATA_ROOT=/cluster/work/cvl/segum/datasets/mot/data/
CMD=./scripts/hpc/octopus/tools/dist_main.sh

echo "Starting job ${JOB_NAME} from ${CONFIG}" 

mkdir -p resources/errors/ 
mkdir -p resources/outputs/

bash ${CMD} \
${GPUS} \
     --config-path ${CONFIG} \
     --outputs-dir ${OUT_DIR} \
     --batch-size ${BS} \
     --lr ${LR} \
     --lr-backbone ${LR_BACKBONE} \
     --lr-points ${LR_POINTS} \
     --data-root ${DATA_ROOT} \
     --pretrained-model /srv/beegfs02/scratch/3d_tracking/data/video_depth/samba/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth