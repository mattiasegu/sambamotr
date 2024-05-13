###############
##### Data preprocessing
BASE_PATH=/srv/beegfs02/scratch/3d_tracking/data/datasets/depth/SportsMOT
TMPDIR=/scratch/yangyun/samba
DATA_ROOT=${TMPDIR}/data  # New data root value
###############

mkdir -p ${DATA_ROOT}/SportsMOT/
cp -r ${BASE_PATH}/annotations ${DATA_ROOT}/SportsMOT/
cp ${BASE_PATH}/train_seqmap.txt ${DATA_ROOT}/SportsMOT/
cp ${BASE_PATH}/val_seqmap.txt ${DATA_ROOT}/SportsMOT/
tar -xf ${BASE_PATH}/train.tar -C ${DATA_ROOT}/SportsMOT/
tar -xf ${BASE_PATH}/val.tar -C ${DATA_ROOT}/SportsMOT/
