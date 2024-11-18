from flask import Flask, render_template, send_from_directory, send_file
import pandas as pd
import os
import os.path as osp

app = Flask(__name__)

# Configuration
EXP_DIR = "pretrained/sambamotr/sportsmot/sambamotr_residual_masking_sync_longer_sched2_lr_0.0002/val/det_0.5_track_0.5_miss_30_interval_1/checkpoint_17_tracker/"
CSV_PATH = osp.join(EXP_DIR, "sorted_pedestrian_detailed.csv")
# VIDEO_DIR = '/BS/diffusion-track/work/sambamotr/outputs/vis/SportsMOT/val'
VIDEO_DIR = '/BS/diffusion-track/work/sambamotr/outputs/vis/SportsMOT/gts/val'
TOP_K = 44
# TOP_K = 10

# Read and parse the CSV
df = pd.read_csv(CSV_PATH)
top_k_seqs = df.head(TOP_K)['seq'].tolist()

@app.route('/')
def index():
    return render_template('index.html', videos=top_k_seqs)

import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/check_video/<seq>')
def check_video(seq):
    video_path = osp.join(VIDEO_DIR, seq, 'output_compressed.mp4')
    if os.path.exists(video_path):
        return f"File exists: {video_path}"
    else:
        return f"File does not exist: {video_path}"

@app.route('/videos/<seq>/output_compressed.mp4')
def serve_video(seq):
    video_path = osp.join(VIDEO_DIR, seq, "output_compressed.mp4")
    logging.debug(f"Checking video path: {video_path}")
    if os.path.exists(video_path):
        logging.debug(f"Serving video from: {video_path}")
        try:
            return send_file(video_path)
        except Exception as e:
            logging.error(f"Error serving video: {e}")
            return "Internal Server Error", 500
    else:
        logging.error(f"File not found: {video_path}")
        return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True)