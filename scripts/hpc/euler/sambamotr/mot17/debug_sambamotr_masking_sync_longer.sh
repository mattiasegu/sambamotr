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

DATASET=mot17
JOB_NAME=sambamotr_residual_masking_sync_longer
CONFIG=./configs/sambamotr/${DATASET}/def_detr/train_residual_masking_sync_longer.yaml

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

CMD=scripts/hpc/euler/sambamotr/mot17/mot17_main.sh

echo "Launching ${CMD} on ${GPUS} gpus."
echo "Starting job ${JOB_NAME} from ${CONFIG}" 

mkdir -p resources/errors/ 
mkdir -p resources/outputs/

     # --gpus=${GPUS_TYPE}:${GPUS_PER_NODE} \
     # --gpus=${GPUS_PER_NODE} \
     # --gres=gpumem:20g \

GPUS_TYPE=rtx_4090  # GPUS_TYPE=(rtx_3090 | rtx_4090 | titan_rtx)
PORT=$(shuf -i 24000-29500 -n 1)
# ID=$(sbatch \
#      --parsable \
#      -t ${TIME} \
#      --job-name=${JOB_NAME} \
#      --ntasks=${GPUS} \
#      --ntasks-per-node=${GPUS_PER_NODE} \
#      --gpus=${GPUS_TYPE}:${GPUS_PER_NODE} \
#      --cpus-per-task=${CPUS_PER_TASK} \
#      --mem-per-cpu ${MEM_PER_CPU} \
#      -e resources/errors/%j.log \
#      -o resources/outputs/%j.log \
#      ${SBATCH_ARGS} \
#      ${CMD} \
#           ${PORT} \
#           --config-path ${CONFIG} \
#           --outputs-dir ${OUT_DIR} \
#           --batch-size ${BS} \
#           --lr ${LR} \
#           --lr-backbone ${LR_BACKBONE} \
#           --lr-points ${LR_POINTS} \
#           --data-root ${DATA_ROOT} \
#           --use-checkpoint \
#           --pretrained-model pretrained/deformable_detr.pth \
#           --launcher slurm \
#           --use-distributed)

module load gcc/8.2.0 python/3.11.2 cuda/11.8.0 nccl/2.16.2-1 cudnn/8.4.0.27

# load venv
WORKSPACE=/cluster/work/cvl/segum/workspaces
export PYTHONPATH=${WORKSPACE}/sambamotr/venv/sambamotr/bin/python
source ${WORKSPACE}/sambamotr/venv/sambamotr/bin/activate


export MASTER_PORT=$PORT
echo "MASTER_PORT="$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "Node rank="$SLURM_NODEID
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE="$SLURM_GPUS_ON_NODE


##############################
##### Data preprocessing
##############################
# bash scripts/hpc/euler/sambamotr/mot17/mot17_copy.sh

DATA_ROOT=${TMPDIR}/data
export MPLBACKEND=Agg

echo ${PYARGS}


NODE_RANK=${NODE_RANK:-0}

GPUS=2
# python -u main.py \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
     main.py \
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
     --use-distributed \
     --available-gpus '0,1'