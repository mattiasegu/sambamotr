import sys
import argparse
import cv2
import time
import os
import numpy as np
import torch
import torchvision.transforms.functional as F
from moviepy.editor import VideoFileClip
from typing import List, Optional, Dict

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from models.utils import load_checkpoint
from models.runtime_tracker import RuntimeTracker
from utils.utils import yaml_to_dict
from utils.nested_tensor import tensor_list_to_nested_tensor
from utils.box_ops import box_cxcywh_to_xyxy
from structures.track_instances import TrackInstances

class Timer:
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

def get_color(idx: int, is_gt: bool = False):
    if is_gt:
        return (0, 255, 0)  # Green for ground truth
    else:
        idx = idx * 3
        return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

def plot_tracking(image, pred_tlwhs, pred_ids, gt_tlwhs, gt_ids, frame_id=0, fps=0., show_text=True, show_ids=True, pred_trajectories=None, gt_trajectories=None, k=10):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    if show_text:
        cv2.putText(im, f'frame: {frame_id} fps: {fps:.2f} pred: {len(pred_tlwhs)} gt: {len(gt_tlwhs)}',
                    (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    # Plot predicted tracks
    for i, tlwh in enumerate(pred_tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(pred_ids[i])
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        if show_ids:
            cv2.putText(im, f'P{obj_id}', (intbox[0], intbox[1]-12), cv2.FONT_HERSHEY_PLAIN,
                        text_scale, color, thickness=text_thickness)
        
        if pred_trajectories is not None and obj_id in pred_trajectories:
            recent_traj = [t[1] for t in pred_trajectories[obj_id] if frame_id - k < t[0] <= frame_id]
            for j in range(len(recent_traj) - 1):
                start = recent_traj[j]
                end = recent_traj[j + 1]
                cv2.line(im, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color=color, thickness=2)

    # Plot ground truth tracks
    for i, tlwh in enumerate(gt_tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(gt_ids[i])
        color = get_color(abs(obj_id), is_gt=True)
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        if show_ids:
            cv2.putText(im, f'G{obj_id}', (intbox[0], intbox[1]-36), cv2.FONT_HERSHEY_PLAIN,
                        text_scale, color, thickness=text_thickness)
        
        if gt_trajectories is not None and obj_id in gt_trajectories:
            recent_traj = [t[1] for t in gt_trajectories[obj_id] if frame_id - k < t[0] <= frame_id]
            for j in range(len(recent_traj) - 1):
                start = recent_traj[j]
                end = recent_traj[j + 1]
                cv2.line(im, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color=color, thickness=2)

    return im

def read_frames_from_folder(folder_path):
    frames = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, file_name)
            frame = cv2.imread(file_path)
            frames.append(frame)
    return frames

def read_ground_truth(gt_path: str) -> Dict[int, List]:
    trajectories = {}
    with open(gt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            try:
                frame_id = int(parts[0])
                obj_id = int(parts[1])
                x1, y1, w, h = map(float, parts[2:6])
                centroid = [(x1 + w / 2), (y1 + h / 2)]
                if obj_id not in trajectories:
                    trajectories[obj_id] = []
                trajectories[obj_id].append((frame_id, centroid, [x1, y1, w, h]))
            except ValueError:
                continue
    return trajectories

def demo_processing(
        model_path: str,
        config_path: str,
        input_path: str,
        gt_path: str,
        out_dir: str,
        show_text: bool,
        with_timestamp: bool,
        fps: int = 30,
        show_ids: bool = True,
        k: int = 10,
        save_frames: bool = False,
):
    # Load model
    config = yaml_to_dict(config_path)
    model = build_model(config)
    load_checkpoint(model=model, path=model_path)
    model.eval()
    print("Model loaded.")

    # Prepare output directory
    save_folder = out_dir
    if with_timestamp:
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        save_folder = os.path.join(save_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)

    # Read input frames
    if os.path.isdir(input_path):
        frames = read_frames_from_folder(input_path)
        assert len(frames) > 0, "The input directory is empty."
    else:
        cap = cv2.VideoCapture(input_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    height, width, _ = frames[0].shape

    # Prepare video writer
    save_path = os.path.join(save_folder, "output.mp4")
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Initialize tracker and other variables
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
    pred_trajectories = {}
    gt_trajectories = read_ground_truth(gt_path)

    with torch.no_grad():
        for frame in frames:
            if frame_id % 20 == 0:
                print(f'Processing frame {frame_id} ({1. / max(1e-5, timer.average_time):.2f} fps)')
            
            image, ori_image = process_image(frame)
            frame_tensor = tensor_list_to_nested_tensor([image]).to("cuda")
            timer.tic()
            frame_id += 1
            
            # Model inference
            res = model(frame=frame_tensor, tracks=tracks)
            previous_tracks, new_tracks = tracker.update(model_outputs=res, tracks=tracks)
            tracks: List[TrackInstances] = model.postprocess_single_frame(previous_tracks, new_tracks, None, intervals=[1])
            tracks_result = tracks[0].to(torch.device("cpu"))
            
            # Post-processing
            tracks_result.area = tracks_result.boxes[:, 2] * width * tracks_result.boxes[:, 3] * height
            tracks_result = filter_by_score(tracks_result, thresh=0.5)
            tracks_result = filter_by_area(tracks_result)
            tracks_result.boxes = box_cxcywh_to_xyxy(tracks_result.boxes)
            tracks_result.boxes = (tracks_result.boxes * torch.as_tensor([width, height, width, height], dtype=torch.float))
            
            # Prepare predicted tracks
            pred_tlwhs, pred_ids = [], []
            for i in range(len(tracks_result)):
                x1, y1, x2, y2 = tracks_result.boxes[i].tolist()
                w, h = x2 - x1, y2 - y1
                pred_tlwhs.append([x1, y1, w, h])
                pred_ids.append(tracks_result.ids[i].item())
                obj_id = tracks_result.ids[i].item()
                centroid = [(x1 + x2) / 2, (y1 + y2) / 2]
                if obj_id not in pred_trajectories:
                    pred_trajectories[obj_id] = []
                pred_trajectories[obj_id].append((frame_id, centroid))

            # Prepare ground truth tracks for the current frame
            gt_tlwhs, gt_ids = [], []
            for obj_id, trajectory in gt_trajectories.items():
                for t in trajectory:
                    if t[0] == frame_id:
                        gt_tlwhs.append(t[2])
                        gt_ids.append(obj_id)
                        break

            timer.toc()

            # Plot tracking results
            online_im = plot_tracking(
                ori_image, pred_tlwhs, pred_ids, gt_tlwhs, gt_ids,
                frame_id=frame_id, fps=1. / timer.average_time,
                show_text=show_text, show_ids=show_ids,
                pred_trajectories=pred_trajectories, gt_trajectories=gt_trajectories, k=k
            )
            
            vid_writer.write(online_im)

            # Save individual frames if the option is enabled
            if save_frames:
                frame_save_path = os.path.join(save_folder, f"frame_{frame_id:04d}.jpg")
                cv2.imwrite(frame_save_path, online_im)

    vid_writer.release()
    return save_path

def main():
    parser = argparse.ArgumentParser(description="Generate a video demo comparing ground truth and predicted tracklets.")
    parser.add_argument('in_video_path', type=str, help="Path to the input video file or directory containing frames.")
    parser.add_argument('gt_path', type=str, help="Path to the ground truth file.")
    parser.add_argument('output_dir', type=str, help="Path to the output directory.")
    parser.add_argument('config_path', type=str, help="Path to the model configuration file.")
    parser.add_argument('model_path', type=str, help="Path to the model checkpoint file.")
    parser.add_argument('--fps', type=int, default=30, help="Video fps, to use in case the in_video_path is a directory.")
    parser.add_argument('--no_timestamp', action='store_true', help="Remove timestamp from the output dir.")
    parser.add_argument('--no_text', action='store_true', help="Remove text from the video.")
    parser.add_argument('--no_ids', action='store_true', help="Remove IDs from the video.")
    parser.add_argument('--k', type=int, default=10, help="Number of recent frames to visualize for trajectories.")
    parser.add_argument('--save_frames', action='store_true', help="Save individual frames as images.")
    parser.add_argument('--create_gif', action='store_true', help="Create a GIF from the output video.")
    
    args = parser.parse_args()
    
    # Process the video
    output_path = demo_processing(
        model_path=args.model_path,
        config_path=args.config_path,
        input_path=args.in_video_path,
        gt_path=args.gt_path,
        out_dir=args.output_dir,
        show_text=not args.no_text,
        with_timestamp=not args.no_timestamp,
        fps=args.fps,
        show_ids=not args.no_ids,
        k=args.k,
        save_frames=args.save_frames,
    )

    # Define the output paths
    save_folder = os.path.dirname(output_path)
    output_gif_path = os.path.join(save_folder, "output_compressed.gif")

    # Load the video clip
    clip = VideoFileClip(output_path)

    # Convert to GIF if the option is enabled
    if args.create_gif:
        clip.write_gif(output_gif_path, fps=10, program='imageio', opt='nq')

    print(f"Output video saved to: {output_path}")
    if args.create_gif:
        print(f"Output GIF saved to: {output_gif_path}")

if __name__ == "__main__":
    main()