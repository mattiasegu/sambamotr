# @Author       : Ruopeng Gao
# @Date         : 2022/11/21

import os
import yaml

from torch.utils import tensorboard as tb

from utils.utils import yaml_to_dict


def evaluate(config: dict):
    eval_split = config["EVAL_DATA_SPLIT"]
    eval_dir = config["EVAL_DIR"]
    exp_name = config["EXP_NAME"] if "EXP_NAME" in config else ""
    interval = config["EVAL_INTERVAL"] if "EVAL_INTERVAL" in config else 1
    det_score_thresh = config["DET_SCORE_THRESH"] 
    track_score_thresh = config["TRACK_SCORE_THRESH"] 
    result_score_thresh = config["RESULT_SCORE_THRESH"] 
    update_thresh = config["UPDATE_THRESH"] 
    miss_tolerance = config["MISS_TOLERANCE"] 

    if config["EVAL_PORT"] is not None:
        port = config["EVAL_PORT"]
    else:
        port = 22701
    outputs_dir = os.path.join(eval_dir, eval_split)
    os.makedirs(outputs_dir, exist_ok=True)
    eval_states_path = os.path.join(outputs_dir, "eval_states.yaml")
    if os.path.exists(eval_states_path):
        eval_states: dict = yaml_to_dict(eval_states_path)
    else:
        eval_states: dict = {
            "NEXT_INDEX": 0,
        }
    # Tensorboard Setting
    tb_writer = tb.SummaryWriter(
        log_dir=os.path.join(outputs_dir, "tb")
    )

    if config["EVAL_MODE"] == "specific":
        if config["EVAL_MODEL"] is None:
            raise ValueError("--eval-model should not be None.")
        metrics = eval_model(model=config["EVAL_MODEL"], eval_dir=eval_dir, exp_name=exp_name,
                             data_root=config['DATA_ROOT'], dataset_name=config["DATASET"], data_split=eval_split,
                             threads=config["EVAL_THREADS"], port=port, config_path=config["CONFIG_PATH"], interval=interval,
                             det_score_thresh=det_score_thresh, track_score_thresh=track_score_thresh, result_score_thresh=result_score_thresh,
                             update_thresh=update_thresh, miss_tolerance=miss_tolerance)
    elif config["EVAL_MODE"] == "continue":
        init_index = eval_states["NEXT_INDEX"]
        for i in range(init_index, 10000):
            model = "checkpoint_" + str(i) + ".pth"
            if os.path.exists(os.path.join(eval_dir, model)):
                if os.path.exists(os.path.join(eval_dir, eval_split, model.split(".")[0] + "_tracker",
                                               "pedestrian_summary.txt")):
                    pass
                else:
                    metrics = eval_model(
                        model=model, eval_dir=eval_dir, exp_name=exp_name,
                        data_root=config["DATA_ROOT"], dataset_name=config["DATASET"], data_split=eval_split,
                        threads=config["EVAL_THREADS"], port=port, config_path=config["CONFIG_PATH"], interval=interval,
                             det_score_thresh=det_score_thresh, track_score_thresh=track_score_thresh, result_score_thresh=result_score_thresh,
                             update_thresh=update_thresh, miss_tolerance=miss_tolerance)
                    metrics_to_tensorboard(writer=tb_writer, metrics=metrics, epoch=i)
                eval_states["NEXT_INDEX"] = i + 1
                with open(eval_states_path, mode="w") as f:
                    yaml.dump(eval_states, f, allow_unicode=True)
    else:
        raise ValueError(f"Eval mode '{config['EVAL_MODE']}' is not supported.")

    with open(eval_states_path, mode="w") as f:
        yaml.dump(eval_states, f, allow_unicode=True)

    return


def eval_model(model: str, eval_dir: str, data_root: str, dataset_name: str, data_split: str, threads: int, port: int,
               config_path: str, exp_name: str, interval: int, det_score_thresh: float, track_score_thresh: float,
               result_score_thresh: float, update_thresh: float, miss_tolerance: float):
    print(f"===>  Running checkpoint '{model}'")

    if threads > 1:
        os.system(f"python -m torch.distributed.run --nproc_per_node={str(threads)} --master_port={port} "
                  f"main.py --mode submit --submit-dir {eval_dir} --exp-name {exp_name} --submit-model {model} "
                  f"--data-root {data_root} --submit-data-split {data_split} --eval-interval {interval} "
                  f"--use-distributed --config-path {config_path} "
                  f"--det-score-thresh {det_score_thresh} --track-score-thresh {track_score_thresh} "
                  f"--result-score-thresh {result_score_thresh} --update-thresh {update_thresh} --miss-tolerance {miss_tolerance}")
    else:
        os.system(f"python main.py --mode submit --submit-dir {eval_dir} --exp-name {exp_name} --submit-model {model} "
                  f"--data-root {data_root} --submit-data-split {data_split} --config-path {config_path} --eval-interval {interval} "
                  f"--det-score-thresh {det_score_thresh} --track-score-thresh {track_score_thresh} "
                  f"--result-score-thresh {result_score_thresh} --update-thresh {update_thresh} --miss-tolerance {miss_tolerance}")

    # 将结果移动到对应的文件夹
    tracker_dir = os.path.join(eval_dir, data_split, exp_name, "tracker")
    tracker_mv_dir = os.path.join(eval_dir, data_split, exp_name, model.split(".")[0] + "_tracker")
    
    if os.path.exists(tracker_mv_dir):
        import shutil
        shutil.rmtree(tracker_mv_dir)
    os.system(f"mv {tracker_dir} {tracker_mv_dir}")

    # 进行指标计算
    data_dir = os.path.join(data_root, dataset_name)
    if dataset_name == "DanceTrack" or dataset_name == "SportsMOT":
        gt_dir = os.path.join(data_dir, data_split)
    elif dataset_name == "BFT":
        gt_dir = os.path.join(data_dir, "annotations_mot", data_split)
    elif "MOT17" in dataset_name:
        gt_dir = os.path.join(data_dir, "images", data_split)
    else:
        raise NotImplementedError(f"Eval Engine DO NOT support dataset '{dataset_name}'")
    if dataset_name == "DanceTrack" or dataset_name == "SportsMOT":
        os.system(f"python3 SparseTrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  "
                  f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                  f"--SEQMAP_FILE {os.path.join(data_dir, f'{data_split}_seqmap.txt')} "
                  f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                  f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                  f"--TRACKERS_FOLDER {tracker_mv_dir} --EVAL_INTERVAL {interval}")
    elif dataset_name == "BFT":
        os.system(f"python3 SparseTrackEval/scripts/run_bft.py --SPLIT_TO_EVAL {data_split}  "
                  f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                  f"--GT_LOC_FORMAT {{gt_folder}}/{{seq}}.txt "
                  f"--SEQMAP_FILE {os.path.join(data_dir, f'{data_split}_seqmap.txt')} "
                  f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                  f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                  f"--TRACKERS_FOLDER {tracker_mv_dir} --EVAL_INTERVAL {interval}")
    elif "MOT17" in dataset_name:
        if "mot15" in data_split:
            os.system(f"python3 SparseTrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  "
                      f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                      f"--SEQMAP_FILE {os.path.join(data_dir, f'{data_split}_seqmap.txt')} "
                      f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                      f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                      f"--TRACKERS_FOLDER {tracker_mv_dir} --BENCHMARK MOT15 --EVAL_INTERVAL {interval}")
        else:
            os.system(f"python3 SparseTrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  "
                      f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                      f"--SEQMAP_FILE {os.path.join(data_dir, f'{data_split}_seqmap.txt')} "
                      f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                      f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                      f"--TRACKERS_FOLDER {tracker_mv_dir} --BENCHMARK MOT17 --EVAL_INTERVAL {interval}")
    else:
        raise NotImplementedError(f"Do not support this Dataset name: {dataset_name}")

    metric_path = os.path.join(tracker_mv_dir, "pedestrian_summary.txt")
    with open(metric_path) as f:
        metric_names = f.readline()[:-1].split(" ")
        metric_values = f.readline()[:-1].split(" ")
    metrics = {
        n: float(v) for n, v in zip(metric_names, metric_values)
    }
    return metrics


def metrics_to_tensorboard(writer: tb.SummaryWriter, metrics: dict, epoch: int):
    for k, v in metrics.items():
        writer.add_scalar(tag=k, scalar_value=v, global_step=epoch)
    return
