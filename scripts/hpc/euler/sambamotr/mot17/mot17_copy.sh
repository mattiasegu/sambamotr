#!/usr/bin/env bash
###############
##### Data preprocessing
DATA_ROOT=${TMPDIR}/data  # New data root value
ROOT_DIR=/cluster/work/cvl/segum/datasets/mot/data



# MOT17
mkdir -p ${DATA_ROOT}/MOT17/
mkdir -p ${DATA_ROOT}/MOT17/images/

cp -r /MOT17/gts ${DATA_ROOT}/MOT17/
cp ${ROOT_DIR}/MOT17/images/train.tar ${DATA_ROOT}/MOT17/images
cp ${ROOT_DIR}/MOT17/train_seqmap.txt ${DATA_ROOT}/MOT17/

tar -xf ${DATA_ROOT}/MOT17/images/train.tar -C ${DATA_ROOT}/MOT17/images


# MOT15
mkdir -p ${DATA_ROOT}/MOT15/
mkdir -p ${DATA_ROOT}/MOT15/images/

cp -r /MOT15/gts ${DATA_ROOT}/MOT15/
cp ${ROOT_DIR}/MOT15/images/train.tar ${DATA_ROOT}/MOT15/images
cp ${ROOT_DIR}/MOT15/train_seqmap.txt ${DATA_ROOT}/MOT15/

tar -xf ${DATA_ROOT}/MOT15/images/train.tar -C ${DATA_ROOT}/MOT15/images


# CrowdHuman
mkdir -p ${DATA_ROOT}/CrowdHuman/
mkdir -p ${DATA_ROOT}/CrowdHuman/images/

cp /CrowdHuman/annotations_val.odgt ${DATA_ROOT}/CrowdHuman/
cp -r /CrowdHuman/gts ${DATA_ROOT}/CrowdHuman/
cp ${ROOT_DIR}/CrowdHuman/images/val.tar ${DATA_ROOT}/CrowdHuman/images
cp ${ROOT_DIR}/CrowdHuman/train_seqmap.txt ${DATA_ROOT}/CrowdHuman/

tar -xf ${DATA_ROOT}/CrowdHuman/images/val.tar -C ${DATA_ROOT}/CrowdHuman/images

###############