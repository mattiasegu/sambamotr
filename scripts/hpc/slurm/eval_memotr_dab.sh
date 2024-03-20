#!/bin/bash
JOB_NAME=eval_memotr_dancetrack
TIME=24:00:00  # TIME=(24:00:00)
PARTITION=gpu22  # PARTITION=(gpu16 | gpu20 | gpu22) 
GPUS=1
CPUS=16
MEM_PER_CPU=22000
SBATCH_ARGS=${SBATCH_ARGS:-""}
GPUS_PER_NODE=${GPUS}
CPUS_PER_TASK=${CPUS}

CONFIG=./configs/train_dancetrack.yaml
OUT_DIR=./outputs/memotr_dancetrack/
BS=1
DATA_ROOT=/BS/diffusion-track/nobackup/data/mot/

if [ $GPUS -gt 1 ]
then
     CMD=tools/dist_main.sh
else
     CMD=tools/main.sh
fi

# for ID in 0 1 2 3 4 5 6 7 8; do
#      MODEL_NAME=checkpoint_${ID}.pth
#      ID=$(sbatch \
#           --parsable \
#           -t ${TIME} \
#           --job-name=${JOB_NAME} \
#           -p ${PARTITION} \
#           --gres=gpu:${GPUS_PER_NODE} \
#           -e resources/errors/%j.log \
#           -o resources/outputs/%j.log \
#           ${SBATCH_ARGS} \
#      ${CMD} \
#           ${GPUS} \
#           --mode eval \
#           --eval-mode specific \
#           --data-root ${DATA_ROOT} \
#           --eval-dir ${OUT_DIR} \
#           --eval-model ${MODEL_NAME} \
#           --eval-threads ${GPUS})
# done


MODEL_NAME=checkpoint_0.pth
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
     --mode eval \
     --eval-mode continue \
     --data-root ${DATA_ROOT} \
     --eval-dir ${OUT_DIR} \
     --eval-model ${MODEL_NAME} \
     --eval-threads ${GPUS})