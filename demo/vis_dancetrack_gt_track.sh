#!/bin/bash

# Call on DanceTrack:
# ./demo/vis_dancetrack.sh \
#     /BS/diffusion-track/nobackup/data/mot/DanceTrack/test/ \
#     outputs/vis/DanceTrack/test/ \
#     pretrained/sambamotr/dancetrack/sambamotr_residual_masking_sync_longer_lr_0.0002/train/config.yaml \
#     pretrained/sambamotr/dancetrack/sambamotr_residual_masking_sync_longer_lr_0.0002/checkpoint_14.pth \
#     --no_text  \
#     --no_timestamp  \
#     --fps=20

# Example of single video call:
# python demo/demo_track.py \
#     /BS/diffusion-track/nobackup/data/mot/DanceTrack/test/dancetrack0038/img1/ \
#     outputs/vis/DanceTrack/gts/dancetrack0038/ \
#     pretrained/sambamotr/dancetrack/sambamotr_residual_masking_sync_longer_lr_0.0002/train/config.yaml \
#     pretrained/sambamotr/dancetrack/sambamotr_residual_masking_sync_longer_lr_0.0002/checkpoint_14.pth \
#     --no_text \
#     --no_timestamp \
#     --no_ids \
#     --fps 20 \ 
#     --k 10 \
#     --save_frames

python demo/demo_gt_track.py \
    /BS/diffusion-track/nobackup/data/mot/DanceTrack/val/dancetrack0030/img1/ \
    /BS/diffusion-track/nobackup/data/mot/DanceTrack/val/dancetrack0030/gt/gt.txt \
    outputs/vis/DanceTrack/gts/dancetrack0030/ \
    --no_text \
    --no_timestamp \
    --no_ids \
    --fps 20 \
    --k 10 \
    # --save_frames
# 

# Check if the correct number of arguments is provided
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <input_base_dir> <output_base_dir> <config_path> <model_path> [--no_text] [--no_timestamp] [--fps]"
    exit 1
fi

INPUT_BASE_DIR=$1
OUTPUT_BASE_DIR=$2
CONFIG_PATH=$3
MODEL_PATH=$4
NO_TEXT=$5
NO_TIMESTAMP=$6
FPS=$7

# Loop through all subdirectories in the input base directory
for SUBDIR in "$INPUT_BASE_DIR"/*; do
    if [ -d "$SUBDIR" ]; then
        SUBDIR_NAME=$(basename "$SUBDIR")
        INPUT_PATH="$SUBDIR/img1"
        OUTPUT_DIR="$OUTPUT_BASE_DIR/$SUBDIR_NAME"
        
        echo "Processing directory: $SUBDIR_NAME"
        echo "Input path: $INPUT_PATH"
        echo "Output directory: $OUTPUT_DIR"
        
        # Create the output directory if it doesn't exist
        mkdir -p "$OUTPUT_DIR"

        # Run the Python script for the current subdirectory
        python demo/demo.py "$INPUT_PATH" "$OUTPUT_DIR" "$CONFIG_PATH" "$MODEL_PATH" "$NO_TEXT" "$NO_TIMESTAMP" "$FPS"
    fi
done
