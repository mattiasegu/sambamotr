#!/bin/bash
module load gcc/8.2.0
module load python/3.11.2
module load cuda/11.8.0
module load nccl/2.16.2-1
module load cudnn/8.4.0.27

WORKSPACE=$1  # e.g. /cluster/work/cvl/segum/workspaces

# python -m venv --system-site-packages ${WORKSPACE}/motr/venv/motr
python -m venv ${WORKSPACE}/sambamotr/venv/sambamotr
export PYTHONPATH=${WORKSPACE}/sambamotr/venv/sambamotr/bin/python
source ${WORKSPACE}/sambamotr/venv/sambamotr/bin/activate

pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib pyyaml scipy tqdm tensorboard einops
pip install opencv-python

# From https://github.com/fundamentalvision/Deformable-DETR
cd ./models/ops/
# Build for different CUDA architectures (refer to https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.7 8.9" sh make.sh