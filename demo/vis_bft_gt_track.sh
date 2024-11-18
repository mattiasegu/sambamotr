#!/bin/bash

# # Call on BFT:
# ./demo/vis_bft_gt_track.sh \
#     /BS/diffusion-track/nobackup/data/mot/BFT/test/ \
#     /BS/diffusion-track/nobackup/data/mot/BFT/annotations_mot/test/ \
#     outputs/vis/BFT/gts/test/ \
#     --no_text  \
#     --no_timestamp  \
#     --no_ids  \
#     --fps=20 \
#     --k=100 \
#     --save_frames

# Example of single video call:
python demo/demo_gt_track.py \
    /BS/diffusion-track/nobackup/data/mot/BFT/test/An1002/ \
    /BS/diffusion-track/nobackup/data/mot/BFT/annotations_mot/test/An1002.txt \
    outputs/vis/BFT/gts_rainbow/test/An1002 \
    --no_text \
    --no_timestamp \
    --no_ids \
    --fps 20 \
    --k 10 \
    --save_frames \
    --thickness 2

python demo/demo_gt_and_pred_track.py \
    /BS/diffusion-track/nobackup/data/mot/BFT/test/An1002/ \
    /BS/diffusion-track/nobackup/data/mot/BFT/annotations_mot/test/An1002.txt \
    outputs/vis/BFT/gts_rainbow/test/An1002 \
    pretrained/sambamotr/bft/sambamotr_residual_masking_sync_longer_lr_0.0002/train/config.yaml \
    pretrained/sambamotr/bft/sambamotr_residual_masking_sync_longer_lr_0.0002/checkpoint_19.pth \
    --no_text \
    --no_timestamp \
    --no_ids \
    --fps 20 \
    --k 10 \
    --save_frames 

# python demo/demo_gt_track.py \
#     /BS/diffusion-track/nobackup/data/mot/DanceTrack/val/dancetrack0030/img1/ \
#     /BS/diffusion-track/nobackup/data/mot/DanceTrack/val/dancetrack0030/gt/gt.txt \
#     outputs/vis/DanceTrack/gts/dancetrack0030/ \
#     --no_text \
#     --no_timestamp \
#     --no_ids \
#     --fps 20 \
#     --k 10 \
    # --save_frames
# 

# Check if the correct number of arguments is provided
if [ "$#" -ne 10 ]; then
    echo "Usage: $0 <input_base_dir> <output_base_dir> <config_path> <model_path> [--no_text] [--no_timestamp] [--fps]"
    exit 1
fi

INPUT_BASE_DIR=$1
GT_BASE_DIR=$2
OUTPUT_BASE_DIR=$3
NO_TEXT=$4
NO_TIMESTAMP=$5
NO_IDS=$6
FPS=$7
K=$8
SAVE_FRAMES=$9

# Loop through all subdirectories in the input base directory
for SUBDIR in "$INPUT_BASE_DIR"/*; do
    if [ -d "$SUBDIR" ]; then
        SUBDIR_NAME=$(basename "$SUBDIR")
        INPUT_PATH="$SUBDIR"
        GT_PATH="$GT_BASE_DIR/$SUBDIR_NAME.txt"
        OUTPUT_DIR="$OUTPUT_BASE_DIR/$SUBDIR_NAME"
        
        echo "Processing directory: $SUBDIR_NAME"
        echo "Input path: $INPUT_PATH"
        echo "Gt path: $GT_PATH"
        echo "Output directory: $OUTPUT_DIR"
        
        # Create the output directory if it doesn't exist
        mkdir -p "$OUTPUT_DIR"

        # Run the Python script for the current subdirectory
        python demo/demo_gt_track.py "$INPUT_PATH" "$GT_PATH" "$OUTPUT_DIR" "$NO_TEXT" "$NO_TIMESTAMP" "$NO_IDS" "$FPS" "$K" "$SAVE_FRAMES"
    fi
done
