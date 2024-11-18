#!/bin/bash

# # # Call on SportsMOT:
# ./demo/vis_sportsmot_gt_track.sh \
#     /BS/diffusion-track/nobackup/data/mot/SportsMOT/val/ \
#     outputs/vis/SportsMOT/gts_rainbow/val/ \
#     --no_text  \
#     --no_timestamp  \
#     --no_ids  \
#     --fps=25 \
#     --k=50 \
#     --save_frames \
#     --thickness=6 \
#     --smooth_trajectory=20

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


# python demo/demo_gt_track.py \
#     /BS/diffusion-track/nobackup/data/mot/SportsMOT/val/v_0kUtTtmLaJA_c010/img1/ \
#     /BS/diffusion-track/nobackup/data/mot/SportsMOT/val/v_0kUtTtmLaJA_c010/gt/gt.txt \
#     outputs/vis/SportsMOT/gts_rainbow/v_0kUtTtmLaJA_c010/ \
#     --no_text \
#     --no_timestamp \
#     --no_ids \
#     --fps 25 \
#     --k 50 \
#     --save_frames \
#     --thickness 6 \
#     --smooth_trajectory 20

# Check if the correct number of arguments is provided
if [ "$#" -ne 10 ]; then
    echo "Usage: $0 <input_base_dir> <output_base_dir> <config_path> <model_path> [--no_text] [--no_timestamp] [--fps]"
    exit 1
fi

INPUT_BASE_DIR=$1
OUTPUT_BASE_DIR=$2
NO_TEXT=$3
NO_TIMESTAMP=$4
NO_IDS=$5
FPS=$6
K=$7
SAVE_FRAMES=$8
THICK=$9
SMOOTH=${10}

# Loop through all subdirectories in the input base directory
for SUBDIR in "$INPUT_BASE_DIR"/*; do
    if [ -d "$SUBDIR" ]; then
        SUBDIR_NAME=$(basename "$SUBDIR")
        INPUT_PATH="$SUBDIR/img1"
        GT_PATH="$SUBDIR/gt/gt.txt"
        OUTPUT_DIR="$OUTPUT_BASE_DIR/$SUBDIR_NAME"
        
        echo "Processing directory: $SUBDIR_NAME"
        echo "Input path: $INPUT_PATH"
        echo "Gt directory: $GT_PATH"
        echo "Output directory: $OUTPUT_DIR"
        
        # Create the output directory if it doesn't exist
        mkdir -p "$OUTPUT_DIR"

        # Run the Python script for the current subdirectory
        echo "Running demo_gt_track.py with arguments:"
        echo "$INPUT_PATH" "$GT_PATH" "$OUTPUT_DIR" "$NO_TEXT" "$NO_TIMESTAMP" "$NO_IDS" "$FPS" "$K" "$SAVE_FRAMES" "$THICK" "$SMOOTH"
        python demo/demo_gt_track.py "$INPUT_PATH" "$GT_PATH" "$OUTPUT_DIR" "$NO_TEXT" "$NO_TIMESTAMP" "$NO_IDS" "$FPS" "$K" "$SAVE_FRAMES" "$THICK" "$SMOOTH"
    fi
done
