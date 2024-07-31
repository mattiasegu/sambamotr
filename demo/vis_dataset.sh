#!/bin/bash

# Call on SportsMOT:
# ./demo/vis_dataset.sh \
#     /BS/diffusion-track/nobackup/data/mot/SportsMOT/val/ \
#     outputs/vis/SportsMOT/val/ \
#     pretrained/sambamotr/sportsmot/sambamotr_residual_masking_sync_longer_sched2_lr_0.0002/train/config.yaml \
#     pretrained/sambamotr/sportsmot/sambamotr_residual_masking_sync_longer_sched2_lr_0.0002/checkpoint_17.pth \
#     --no_text  \
#     --no_timestamp  \
#     --fps=25 \
#     img1

# Call on BFT:
# ./demo/vis_dataset.sh \
#     /BS/diffusion-track/nobackup/data/mot/BFT/test/ \
#     outputs/vis/BFT/test/ \
#     pretrained/sambamotr/bft/sambamotr_residual_masking_sync_longer_lr_0.0002/train/config.yaml \
#     pretrained/sambamotr/bft/sambamotr_residual_masking_sync_longer_lr_0.0002/checkpoint_19.pth \
#     --no_text  \
#     --no_timestamp  \
#     --fps=25 \
#      

# Call on DanceTrack:
# ./demo/vis_dataset.sh \
#     /BS/diffusion-track/nobackup/data/mot/DanceTrack/test/ \
#     outputs/vis/DanceTrack/test/ \
#     pretrained/sambamotr/dancetrack/sambamotr_residual_masking_sync_longer_lr_0.0002/train/config.yaml \
#     pretrained/sambamotr/dancetrack/sambamotr_residual_masking_sync_longer_lr_0.0002/checkpoint_14.pth \
#     --no_text  \
#     --no_timestamp  \
#     --fps=20 \
#     img1

# Check if the correct number of arguments is provided
if [ "$#" -ne 8 ]; then
    echo "Usage: $0 <input_base_dir> <output_base_dir> <config_path> <model_path> [--no_text] [--no_timestamp] [--fps] [suffix]"
    exit 1
fi

INPUT_BASE_DIR=$1
OUTPUT_BASE_DIR=$2
CONFIG_PATH=$3
MODEL_PATH=$4
NO_TEXT=$5
NO_TIMESTAMP=$6
FPS=$7
SUFFIX=$8

# Loop through all subdirectories in the input base directory
for SUBDIR in "$INPUT_BASE_DIR"/*; do
    if [ -d "$SUBDIR" ]; then
        SUBDIR_NAME=$(basename "$SUBDIR")
        INPUT_PATH="$SUBDIR/$SUFFIX"
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
