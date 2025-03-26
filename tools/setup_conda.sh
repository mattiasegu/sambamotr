#!/bin/bash

conda create -n sambamotr -y python=3.11  # create a virtual env
conda activate sambamotr               # activate the env
conda install -y pytorch==2.5.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y matplotlib pyyaml scipy tqdm tensorboard einops
pip install opencv-python

cd ./models/ops/
TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.7 8.9" sh make.sh
python test.py