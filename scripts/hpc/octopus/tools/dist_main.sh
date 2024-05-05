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
BASE_PATH=/srv/beegfs02/scratch/3d_tracking/data/datasets/depth/DanceTrack
TMPDIR=/scratch/yangyun/samba
DATA_ROOT=${TMPDIR}/data  # New data root value
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

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export PYTHONFAULTHANDLER=1
# export CUDA_LAUNCH_BLOCKING=1
export MPLBACKEND=Agg

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    main.py \
    --use-distributed \
    $NEW_PYARGS