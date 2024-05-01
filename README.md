# SambaMOTR

The official implementation of [SambaMOTR]().

Authors: [Mattia Segu](https://mattiasegu.github.io).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/memotr-long-term-memory-augmented-transformer/multi-object-tracking-on-dancetrack)](https://paperswithcode.com/sota/multi-object-tracking-on-dancetrack?p=memotr-long-term-memory-augmented-transformer)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/memotr-long-term-memory-augmented-transformer/multiple-object-tracking-on-sportsmot)](https://paperswithcode.com/sota/multiple-object-tracking-on-sportsmot?p=memotr-long-term-memory-augmented-transformer)

![MeMOTR](./assets/overview.png)

**SambaMOTR** is a fully-end-to-end long-term multi-object tracker based on synchronized state space models. Without bells and whistles, we combine the MOTR framework with a Samba (Synchronized Mamba) module to model long-term relationships within and across tracklets, thus significantly improving the association performance.

![Dance Demo](assets/dancetrack_demo.gif)

## News :fire:

- 2024.05.30: We release the main code. More configurations, scripts and checkpoints will be released soon :soon:.

## Installation

### Install with conda
```shell
conda create -n sambamotr -y python=3.11  # create a virtual env
conda activate sambamotr               # activate the env
conda install -y pytorch==2.1.2 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y matplotlib pyyaml scipy tqdm tensorboard einops
pip install opencv-python
```

### Install with venv (alternative)
```shell
python -m venv venv/sambamotr
export PYTHONPATH=venv/sambamotr/bin/python
source venv/sambamotr/bin/activate

pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib pyyaml scipy tqdm tensorboard einops
pip install opencv-python
```

### Build Deformable Attention CUDA ops

You also need to compile the Deformable Attention CUDA ops:

```shell
# From https://github.com/fundamentalvision/Deformable-DETR
cd ./models/ops/
# Build for different CUDA architectures (refer to https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
# TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.7 8.9" sh make.sh
TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6" sh make.sh
# You can test this ops if you need:
python test.py
```

## Data

You should put the unzipped MOT17 and CrowdHuman datasets into the `DATADIR/MOT17/images/` and `DATADIR/CrowdHuman/images/`, respectively. And then generate the ground truth files by running the corresponding script: [./data/gen_mot17_gts.py](./data/gen_mot17_gts.py) and [./data/gen_crowdhuman_gts.py](./data/gen_crowdhuman_gts.py). 

Finally, you should get the following dataset structure:
```
DATADIR/
  ├── DanceTrack/
  │ ├── train/
  │ ├── val/
  │ ├── test/
  │ ├── train_seqmap.txt
  │ ├── val_seqmap.txt
  │ └── test_seqmap.txt
  ├── SportsMOT/
  │ ├── train/
  │ ├── val/
  │ ├── test/
  │ ├── train_seqmap.txt
  │ ├── val_seqmap.txt
  │ └── test_seqmap.txt
  ├── MOT17/
  │ ├── images/
  │ │ ├── train/     # unzip from MOT17
  │ │ └── test/      # unzip from MOT17
  │ └── gts/
  │   └── train/     # generate by ./data/gen_mot17_gts.py
  └── CrowdHuman/
    ├── images/
    │ ├── train/     # unzip from CrowdHuman
    │ └── val/       # unzip from CrowdHuman
    └── gts/
      ├── train/     # generate by ./data/gen_crowdhuman_gts.py
      └── val/       # generate by ./data/gen_crowdhuman_gts.py
```


## Pretrain (Deformable DETR)

We initialize our model with the official Deformable-DETR (with R50 backbone) weights pretrained on the COCO dataset, you can also download the checkpoint we used [here](https://drive.google.com/file/d/1JYKyRYzUH7uo9eVfDaVCiaIGZb5YTCuI/view?usp=sharing). And then put the checkpoint at `pretrained/deformable_detr.pth`.

## Pretrain (DAB-DETR)

We initialize our model with the official DAB-Deformable-DETR (with R50 backbone) weights pretrained on the COCO dataset, you can also download the checkpoint we used [here](https://drive.google.com/file/d/17FxIGgIZJih8LWkGdlIOe9ZpVZ9IRxSj/view?usp=sharing). And then put the checkpoint at `pretrained/dab_deformable_detr.pth`.

## Scripts

### Training
Train SambaMOTR with 8 GPUs on `${DATASET}` (one of `[DanceTrack, SportsMOT, MOT17]`) (recommended to use GPUs with >= 32 GB Memory, like V100-32GB or some else):
```shell
python -m torch.distributed.run --nproc_per_node=8 main.py --use-distributed --config-path ./configs/sambamotr/${DATASET}/def_detr/train_masking_sync.yaml --outputs-dir ./outputs/sambamotr/${DATASET}/ --batch-size 1 --data-root <your data dir path>
```
if your GPU's memory is less than 32 GB, use the flag `--use-checkpoint` to activate [gradient checkpointing](https://pytorch.org/docs/1.13/checkpoint.html?highlight=checkpoint#torch.utils.checkpoint.checkpoint) and reduce the allocated GPU memory. 
```shell
python -m torch.distributed.run --nproc_per_node=8 main.py --use-distributed --config-path ./configs/sambamotr/${DATASET}/def_detr/train_masking_sync.yaml --outputs-dir ./outputs/sambamotr/${DATASET}/ --batch-size 1 --data-root <your data dir path> --use-checkpoint
```

### Submit and Evaluation
You can use this script to evaluate the trained model on the DanceTrack val set:
```shell
python main.py --mode eval --data-root <your data dir path> --eval-mode specific --eval-model <filename of the checkpoint> --eval-dir ./outputs/sambamotr/${DATASET}/ --eval-threads <your gpus num>
```
for submitting, you can use the following scripts:
```shell
python -m torch.distributed.run --nproc_per_node=8 main.py --mode submit --submit-dir ./outputs/sambamotr/${DATASET}/ --submit-model <filename of the checkpoint> --use-distributed --data-root <your data dir path>
```
To reproduce our results, you can download our pre-trained checkpoints from [here](https://drive.google.com/file/...) and mode the corresponding one to `./outputs/sambamotr/${DATASET}/` before running the above scripts.


## Results

### Multi-Object Tracking on the DanceTrack test set

| Methods                  | HOTA | DetA | AssA | checkpoint                                                   |
| ------------------------ | ---- | ---- | ---- | ------------------------------------------------------------ |
| MeMOTR                   | 68.5 | 80.5 | 58.4 | [Google Drive](https://drive.google.com/file/d/1_Xh-TDwwDIeacVEywwlYNvyRmhTKB5K2/view?usp=sharing) |
| MeMOTR (Deformable DETR) | 63.4 | 77.0 | 52.3 | [Google Drive](https://drive.google.com/file/d/1B72E6PGhJmtsx5BsEisJ8vXHXvBK1nTD/view?usp=drive_link) |



### Multi-Object Tracking on the SportsMOT test set
*For all experiments, we do not use extra data (like CrowdHuman) for training.*

| Methods                  | HOTA | DetA | AssA | checkpoint                                                   |
| ------------------------ | ---- | ---- | ---- | ------------------------------------------------------------ |
| MeMOTR                   | 70.0 | 83.1 | 59.1 | [Google Drive](https://drive.google.com/file/d/1sZkOi9r5WXk7uopUXQoF0H0t9o5VjUmw/view?usp=drive_link) |
| MeMOTR (Deformable DETR) | 68.8 | 82.0 | 57.8 | [Google Drive](https://drive.google.com/file/d/14aKtLV5t09LrfegP7yiJk2zgb3bvaOND/view?usp=drive_link) |

### Multi-Object Tracking on the MOT17 test set

| Methods | HOTA | DetA | AssA | checkpoint                                                   |
| ------- | ---- | ---- | ---- | ------------------------------------------------------------ |
| MeMOTR  | 58.8 | 59.6 | 58.4 | [Google Drive](https://drive.google.com/file/d/1MPZJfP91Pb1ThnX5dvxZ7tcjDH8t9hew/view?usp=drive_link) |



### Multi-Category Multi-Object Tracking on the BDD100K val set

| Methods | mTETA | mLocA | mAssocA | checkpoint                                                   |
| ------- | ----- | ----- | ------- | ------------------------------------------------------------ |
| MeMOTR  | 53.6  | 38.1  | 56.7    | [Google Drive](https://drive.google.com/file/d/1vmDme7ANVOjwLrvBJasLHJ2EBEoinA-t/view?usp=drive_link) |



## Contact

- Mattia Segu: mattia.segu@gmail.com

## Citation
```bibtex



```

## Acknowledgement

- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [DAB DETR](https://github.com/IDEA-Research/DAB-DETR)
- [MOTR](https://github.com/megvii-research/MOTR)
- [MeMOTR](https://github.com/MCG-NJU/MeMOTR)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
