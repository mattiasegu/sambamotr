#!/bin/bash
module load cuda/12.1 cudnn/8.8.1 nccl/2.17.1

WORKSPACE=$1  # e.g. /ptmp/msegu/workspaces/sambamotr

# python -m venv --system-site-packages ${WORKSPACE}/motr/venv/motr
python -m venv ${WORKSPACE}/venv/sambamotr
export PYTHONPATH=${WORKSPACE}/venv/sambamotr/bin/python
source ${WORKSPACE}/venv/sambamotr/bin/activate

pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib pyyaml scipy tqdm tensorboard einops
pip install opencv-python

# From https://github.com/fundamentalvision/Deformable-DETR
cd ./models/ops/
# Build for different CUDA architectures (refer to https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.7 8.9" sh make.sh
