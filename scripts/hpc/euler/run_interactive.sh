TIME=4:00:00  # TIME=(24:00:00)
# TIME=1:00:00  # TIME=(24:00:00)
GPUS=2
CPUS=8
MEM_PER_CPU=5000
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
SBATCH_ARGS=${SBATCH_ARGS:-""}
GPUS_PER_NODE=${GPUS}
CPUS_PER_TASK=${CPUS}



srun --kill-on-bad-exit=1 --gpus=${GPUS_PER_NODE} --gres=gpumem:20g \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --mem-per-cpu=${MEM_PER_CPU}  \
    --job-name "InteractiveJob" --time=${TIME} --pty bash

# srun --kill-on-bad-exit=1 --gpus=rtx_4090:${GPUS_PER_NODE} --gres=gpumem:20g \
#     --ntasks=${GPUS} \
#     --ntasks-per-node=${GPUS_PER_NODE} \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --mem-per-cpu=${MEM_PER_CPU}  \
#     --job-name "InteractiveJob" --time=${TIME} --pty bash


# srun --kill-on-bad-exit=1 --gpus=rtx_3090:${GPUS_PER_NODE} --gres=gpumem:20g \
#     --ntasks=${GPUS} \
#     --ntasks-per-node=${GPUS_PER_NODE} \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --mem-per-cpu=${MEM_PER_CPU}  \
#     --job-name "InteractiveJob" --time=${TIME} --pty bash