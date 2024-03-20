TIME=1:00:00  # TIME=(24:00:00)
# TIME=1:00:00  # TIME=(24:00:00)
GPUS=2
CPUS=8
MEM_PER_CPU=5000
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
SBATCH_ARGS=${SBATCH_ARGS:-""}
GPUS_PER_NODE=${GPUS}
CPUS_PER_TASK=${CPUS}

PARTITION=gpu22

srun --partition=${PARTITION} --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --mem-per-cpu=${MEM_PER_CPU}  \
    --job-name "InteractiveJob" --time=${TIME} --pty bash
