"""
python demo/demo.py demo/dancer_demo.mp4 demo/vis pretrained/sambamotr/dancetrack/sambamotr_residual_masking_sync_longer_lr_0.0002/train/config.yaml pretrained/sambamotr/dancetrack/sambamotr_residual_masking_sync_longer_lr_0.0002/checkpoint_14.pth --no_text
python demo/demo.py demo/imgs/img1 outputs/vis/dancetrack pretrained/sambamotr/dancetrack/sambamotr_residual_masking_sync_longer_lr_0.0002/train/config.yaml pretrained/sambamotr/dancetrack/sambamotr_residual_masking_sync_longer_lr_0.0002/checkpoint_14.pth --no_text --fps 20
python demo/demo.py /BS/diffusion-track/nobackup/data/mot/DanceTrack/test/dancetrack0054/img1/ outputs/vis/DanceTrack/test/dancetrack0054/ pretrained/sambamotr/dancetrack/sambamotr_residual_masking_sync_longer_lr_0.0002/train/config.yaml pretrained/sambamotr/dancetrack/sambamotr_residual_masking_sync_longer_lr_0.0002/checkpoint_14.pth --no_text --no_timestamp --fps 20
"""
import sys
import argparse
import cv2
import time
import os
import numpy as np
import torch
from base64 import b64encode
import torchvision.transforms.functional as F
from moviepy.editor import VideoFileClip, ImageSequenceClip
from typing import List, Optional

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from models.utils import load_checkpoint, get_model
from models.runtime_tracker import RuntimeTracker
from utils.utils import yaml_to_dict, is_distributed, distributed_world_size, distributed_rank, inverse_sigmoid
from utils.nested_tensor import tensor_list_to_nested_tensor
from utils.box_ops import box_cxcywh_to_xyxy
from structures.track_instances import TrackInstances

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

def play_video(video_path):
    video = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

def process_image(image):
    ori_image = image.copy()
    h, w = image.shape[:2]
    scale = 800 / min(h, w)
    if max(h, w) * scale > 1536:
        scale = 1536 / max(h, w)
    target_h = int(h * scale)
    target_w = int(w * scale)
    image = cv2.resize(image, (target_w, target_h))
    image = F.normalize(F.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image, ori_image

def filter_by_score(tracks: TrackInstances, thresh: float = 0.7):
    keep = torch.max(tracks.scores, dim=-1).values > thresh
    return tracks[keep]

def filter_by_area(tracks: TrackInstances, thresh: int = 100):
    assert len(tracks.area) == len(tracks.ids), f"Tracks' 'area' should have the same dim with 'ids'"
    keep = tracks.area > thresh
    return tracks[keep]

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None, show_text=True):
    im = np.ascontiguousarray(np.copy(image))
    print(im.shape)
    im_h, im_w = im.shape[:2]
    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    text_scale = 2
    text_thickness = 2
    line_thickness = 3
    radius = max(5, int(im_w/140.))
    
    if show_text:
        cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                    (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN,
                    text_scale, (0, 0, 255), thickness=2)
    
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1+w, y1+h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]-12), cv2.FONT_HERSHEY_PLAIN,
                    text_scale, color, thickness=text_thickness)
        if scores is not None and show_text:
            text = 'score: {:.2f}'.format(scores[i])
            cv2.putText(im, text, (intbox[0], intbox[1]-24), cv2.FONT_HERSHEY_PLAIN,
                        text_scale, color, thickness=text_thickness)
        if ids2 is not None:
            color2 = get_color(abs(int(ids2[i])))
            cv2.putText(im, str(ids2[i]), (intbox[0], intbox[1] - 36),
                        cv2.FONT_HERSHEY_PLAIN, text_scale, color2, thickness=text_thickness)
    return im

def read_frames_from_folder(folder_path):
    frames = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, file_name)
            frame = cv2.imread(file_path)
            frames.append(frame)
    return frames

def demo_processing(
        model_path: str,
        config_path: str,
        input_path: str,
        out_dir: str,
        show_text: bool,
        with_timestamp: bool,
        fps: int = 30,
):
    config = yaml_to_dict(config_path)
    model = build_model(config)
    load_checkpoint(
        model=model,
        path=model_path
    )
    model.eval()
    print("Model loaded.")

    save_folder = out_dir
    if with_timestamp is True:
        current_time = time.localtime()
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        save_folder = os.path.join(save_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    
    if os.path.isdir(input_path):
        frames = read_frames_from_folder(input_path)
        assert len(frames) > 0, "The input directory is empty."
        height, width, _ = frames[0].shape
    else:
        cap = cv2.VideoCapture(input_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        fps = cap.get(cv2.CAP_PROP_FPS)
        height, width, _ = frames[0].shape
        cap.release()

    save_path = os.path.join(save_folder, "output.avi")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (int(width), int(height))
    )
    print((int(width), int(height)))
    result_score_thresh = 0.5
    timer = Timer()
    frame_id = 0
    tracks = [TrackInstances(hidden_dim=model.hidden_dim,
                            num_classes=model.num_classes,
                            state_dim=getattr(model.query_updater, "state_dim", 0),
                            expand=getattr(model.query_updater, "expand", 0),
                            num_layers=getattr(model.query_updater, "num_layers", 0),
                            conv_dim=getattr(model.query_updater, "conv_dim", 0),
                            use_dab=config["USE_DAB"]).to("cuda")] 
    tracker = RuntimeTracker(
        det_score_thresh=0.5, 
        track_score_thresh=0.5,
        miss_tolerance=35,
        use_motion=False,
        motion_min_length=0, 
        motion_max_length=0,
        visualize=False, 
        use_dab=config["USE_DAB"],
    )
    with torch.no_grad():
        for frame in frames:
            if frame_id % 20 == 0:
                print('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            image, ori_image = process_image(frame)
            frame_tensor = tensor_list_to_nested_tensor([image]).to("cuda")
            timer.tic()
            frame_id += 1
            res = model(frame=frame_tensor, tracks=tracks)
            previous_tracks, new_tracks = tracker.update(
                model_outputs=res,
                tracks=tracks
            )
            tracks: List[TrackInstances] = model.postprocess_single_frame(previous_tracks, new_tracks, None, intervals=[1])
            tracks_result = tracks[0].to(torch.device("cpu"))
            ori_h, ori_w = height, width
            tracks_result.area = tracks_result.boxes[:, 2] * ori_w * \
                                tracks_result.boxes[:, 3] * ori_h
            tracks_result = filter_by_score(tracks_result, thresh=result_score_thresh)
            tracks_result = filter_by_area(tracks_result)
            tracks_result.boxes = box_cxcywh_to_xyxy(tracks_result.boxes)
            tracks_result.boxes = (tracks_result.boxes * torch.as_tensor([ori_w, ori_h, ori_w, ori_h], dtype=torch.float))
            online_tlwhs, online_ids = [], []
            for i in range(len(tracks_result)):
                x1, y1, x2, y2 = tracks_result.boxes[i].tolist()
                w, h = x2 - x1, y2 - y1
                online_tlwhs.append([x1, y1, w, h])
                online_ids.append(tracks_result.ids[i].item())
            timer.toc()
            if len(online_tlwhs) > 0:
                online_im = plot_tracking(
                    ori_image, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time, show_text=show_text
                )
            else:
                online_im = ori_image
            vid_writer.write(online_im)
    vid_writer.release()
    return save_path

def main():
    parser = argparse.ArgumentParser(description="Generate a video demo using a multi-object tracking model.")
    parser.add_argument('in_video_path', type=str, help="Path to the input video file or directory containing frames.")
    parser.add_argument('output_dir', type=str, help="Path to the output directory.")
    parser.add_argument('config_path', type=str, help="Path to the model configuration file.")
    parser.add_argument('model_path', type=str, help="Path to the model checkpoint file.")
    parser.add_argument('--fps', type=int, default=30, help="Video fps, to use in case the in_video_path is a directory.")
    parser.add_argument('--no_timestamp', action='store_true', help="Remove timestamp from the output dir.")
    parser.add_argument('--no_text', action='store_true', help="Remove text from the video.")
    
    args = parser.parse_args()

    # Play the original video (this will not work in a headless environment)
    # play_video(args.in_video_path)
    
    # Process the video
    output_path = demo_processing(
        model_path=args.model_path,
        config_path=args.config_path,
        input_path=args.in_video_path,
        out_dir=args.output_dir,
        show_text=not args.no_text,
        with_timestamp=not args.no_timestamp,
        fps=args.fps
    )

    # Play the processed video (this will not work in a headless environment)
    # play_video(output_path)

    # Define the output paths
    save_folder = os.path.dirname(output_path)
    output_gif_path = os.path.join(save_folder, "output_compressed.gif")
    output_video_path = os.path.join(save_folder, "output_compressed.mp4")

    # Load the video clip
    clip = VideoFileClip(output_path)

    # Convert to GIF
    clip.write_gif(output_gif_path, fps=10, program='imageio', opt='nq')

    # Convert to compressed MP4 with improved quality
    clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', bitrate="2000k")

    # Remove heavy .avi file (output_path)
    if os.path.exists(output_path):
        os.remove(output_path)

if __name__ == "__main__":
    main()
