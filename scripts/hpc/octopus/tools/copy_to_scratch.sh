###############
##### Data preprocessing
BASE_PATH=/srv/beegfs02/scratch/3d_tracking/data/datasets/depth/DanceTrack
TMPDIR=/scratch/yangyun/samba
DATA_ROOT=${TMPDIR}/data  # New data root value
###############

mkdir -p ${DATA_ROOT}/DanceTrack/
cp -r ${BASE_PATH}/annotations ${DATA_ROOT}/DanceTrack/
cp ${BASE_PATH}/train_seqmap.txt ${DATA_ROOT}/DanceTrack/
cp ${BASE_PATH}/val_seqmap.txt ${DATA_ROOT}/DanceTrack/
tar -xf ${BASE_PATH}/train.tar -C ${DATA_ROOT}/DanceTrack/
tar -xf ${BASE_PATH}/val.tar -C ${DATA_ROOT}/DanceTrack/
