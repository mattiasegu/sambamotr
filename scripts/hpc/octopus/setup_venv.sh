#!/bin/bash

conda create -n samba python=3.11
conda activate samba

TMPDIR=/scratch/yangyun/tmp pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
TMPDIR=/scratch/yangyun/tmp pip install matplotlib pyyaml scipy tqdm tensorboard einops
TMPDIR=/scratch/yangyun/tmp pip install opencv-python

cd ./models/ops; TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" sh make.sh