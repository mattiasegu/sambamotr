#!/bin/bash

conda create -n automotconda python=3.10  # create a virtual env
conda activate automotconda               # activate the env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install matplotlib pyyaml scipy tqdm tensorboard
pip install opencv-python

cd ./models/ops/
sh make.sh