#!/usr/bin/env bash
module load gcc/8.2.0
module load python/3.11.2
module load cuda/11.8.0
module load nccl/2.16.2-1
module load cudnn/8.4.0.27

# load venv
WORKSPACE=/cluster/work/cvl/segum/workspaces
export PYTHONPATH=${WORKSPACE}/sambamotr/venv/sambamotr/bin/python
source ${WORKSPACE}/sambamotr/venv/sambamotr/bin/activate

GPUS=$1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PYARGS=${@:2}

PORT=$(shuf -i 24000-29500 -n 1)
echo "Copying dataset to tmp dir"

###############
##### Data preprocessing
DATA_ROOT=${TMPDIR}/data  # New data root value
mkdir -p ${DATA_ROOT}/DanceTrack/
cp -r /cluster/work/cvl/segum/datasets/mot/data/DanceTrack/annotations ${DATA_ROOT}/DanceTrack/
cp /cluster/work/cvl/segum/datasets/mot/data/DanceTrack/train.tar ${DATA_ROOT}/DanceTrack/
cp /cluster/work/cvl/segum/datasets/mot/data/DanceTrack/train_seqmap.txt ${DATA_ROOT}/DanceTrack/
cp /cluster/work/cvl/segum/datasets/mot/data/DanceTrack/val.tar ${DATA_ROOT}/DanceTrack/
cp /cluster/work/cvl/segum/datasets/mot/data/DanceTrack/val_seqmap.txt ${DATA_ROOT}/DanceTrack/
tar -xf ${DATA_ROOT}/DanceTrack/train.tar -C ${DATA_ROOT}/DanceTrack/
tar -xf ${DATA_ROOT}/DanceTrack/val.tar -C ${DATA_ROOT}/DanceTrack/
###############

###############
##### Argument manipulation (override DATA_ROOT)

# Initialize an empty string for the new arguments
NEW_PYARGS=""

# Flag to skip the next argument
SKIP_NEXT=0

# Loop through the PYARGS
for arg in $PYARGS; do
    # If the next argument should be skipped, just toggle the flag and continue
    if [ $SKIP_NEXT -eq 1 ]; then
        SKIP_NEXT=0
        continue
    fi
    
    # Check if the argument is --data-root
    if [ "$arg" == "--data-root" ]; then
        # Add modified --data-root with new value to NEW_PYARGS
        NEW_PYARGS+="--data-root $DATA_ROOT "
        # Toggle the flag to skip the next argument since we're replacing it
        SKIP_NEXT=1
    else
        # If not --data-root, just add the argument to NEW_PYARGS
        NEW_PYARGS+="$arg "
    fi
done
echo $NEW_PYARGS
###############

export MPLBACKEND=Agg

python main.py \
    $NEW_PYARGS