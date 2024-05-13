#!/usr/bin/env bash
###############
##### Data preprocessing
DATA_ROOT=${TMPDIR}/data  # New data root value
mkdir -p ${DATA_ROOT}/BFT/
cp -r /cluster/work/cvl/segum/datasets/mot/data/BFT/annotations_mot ${DATA_ROOT}/BFT/
cp /cluster/work/cvl/segum/datasets/mot/data/BFT/train.zip ${DATA_ROOT}/BFT/
cp /cluster/work/cvl/segum/datasets/mot/data/BFT/train_seqmap.txt ${DATA_ROOT}/BFT/
cp /cluster/work/cvl/segum/datasets/mot/data/BFT/val.zip ${DATA_ROOT}/BFT/
cp /cluster/work/cvl/segum/datasets/mot/data/BFT/val_seqmap.txt ${DATA_ROOT}/BFT/

mkdir -p ${DATA_ROOT}/BFT/train/
mkdir -p ${DATA_ROOT}/BFT/val/

unzip ${DATA_ROOT}/BFT/train.zip -d ${DATA_ROOT}/BFT/train
unzip ${DATA_ROOT}/BFT/val.zip -d ${DATA_ROOT}/BFT/val
###############