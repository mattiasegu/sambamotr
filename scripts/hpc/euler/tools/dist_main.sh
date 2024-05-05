#!/usr/bin/env bash
module load gcc/8.2.0 python/3.11.2 cuda/11.8.0 nccl/2.16.2-1 cudnn/8.4.0.27

# load venv
source /cluster/project/cvl/lpiccinelli/sambamotr/venv/sambamotr/bin/activate

PYARGS=${@:1}

PORT=$(shuf -i 24000-29500 -n 1)

export MPLBACKEND=Agg

echo ${PYARGS}

srun -c 4 --kill-on-bad-exit=1 python -u main.py ${PYARGS}
    