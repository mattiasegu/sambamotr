import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist

from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from models import build_model
from data import build_dataset, build_sampler, build_dataloader
from utils.utils import labels_to_one_hot, is_distributed, distributed_rank, set_seed, is_main_process, \
    distributed_world_size
from utils.nested_tensor import tensor_list_to_nested_tensor
from models.sambamotr import SambaMOTR
from structures.track_instances import TrackInstances
from models.criterion import build as build_criterion, ClipCriterion
from models.utils import get_model, save_checkpoint, load_checkpoint, link_checkpoint
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils import tensorboard as tb
from log.logger import Logger, ProgressLogger
from log.log import MetricLog
from models.utils import load_pretrained_model
from submit_engine import Submitter
from eval_engine import metrics_to_tensorboard


def train(config: dict):
    train_logger = Logger(logdir=os.path.join(config["OUTPUTS_DIR"], "train"), only_main=True)
    train_logger.show(head="Configs:", log=config)
    train_logger.write(log=config, filename="config.yaml", mode="w")
    train_logger.tb_add_git_version(git_version=config["GIT_VERSION"])

    set_seed(config["SEED"])

    model = build_model(config=config)

    category_id = config["CATEGORY_ID"] if "CATEGORY_ID" in config else 1  # default category is person

    # Load Pretrained Model
    if config["PRETRAINED_MODEL"] is not None:
        model = load_pretrained_model(model, config["PRETRAINED_MODEL"], show_details=False, category_id=category_id)

    # Data process
    dataset_train = build_dataset(config=config, split="train")
    sampler_train = build_sampler(dataset=dataset_train, shuffle=True)
    dataloader_train = build_dataloader(dataset=dataset_train, sampler=sampler_train,
                                        batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"])

    # Criterion
    criterion = build_criterion(config=config)
    criterion.set_device(torch.cuda.current_device())

    # Optimizer
    param_groups, lr_names = get_param_groups(config=config, model=model)
    optimizer = AdamW(params=param_groups, lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    # Scheduler
    if config["LR_SCHEDULER"] == "MultiStep":
        scheduler = MultiStepLR(
            optimizer,
            milestones=config["LR_DROP_MILESTONES"],
            gamma=config["LR_DROP_RATE"]
        )
    elif config["LR_SCHEDULER"] == "Cosine":
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config["EPOCHS"]
        )
    else:
        raise ValueError(f"Do not support lr scheduler '{config['LR_SCHEDULER']}'")

    # Eval
    val_split = config["EVAL_DATA_SPLIT"] if "EVAL_DATA_SPLIT" in config else "val"
    outputs_dir = os.path.join(config["OUTPUTS_DIR"], val_split)
    tb_writer = tb.SummaryWriter(
        log_dir=os.path.join(outputs_dir, "tb")
    )

    # Training states
    train_states = {
        "start_epoch": 0,
        "global_iters": 0
    }

    # Resume
    if config["RESUME"] is not None:
        load_path = config["RESUME"]
        if os.path.exists(load_path):
            if config["RESUME_SCHEDULER"]:
                load_checkpoint(model=model, path=load_path, states=train_states,
                                optimizer=optimizer)
            else:
                load_checkpoint(model=model, path=load_path, states=train_states)
            for _ in range(train_states["start_epoch"]):
                scheduler.step()
        else:
            train_logger.show(head=f"{load_path} does not exist. Initializing model from scratch.")
            train_logger.write(head=f"{load_path} does not exist. Initializing model from scratch.")

    # Set start epoch
    start_epoch = train_states["start_epoch"]

    if is_distributed():
        model = DDP(module=model, device_ids=[int(os.environ['LOCAL_RANK'])], find_unused_parameters=False)

    multi_checkpoint = "MULTI_CHECKPOINT" in config and config["MULTI_CHECKPOINT"]
    
    # Training:
    for epoch in range(start_epoch, config["EPOCHS"]):
        if is_distributed():
            sampler_train.set_epoch(epoch)
        dataset_train.set_epoch(epoch)

        sampler_train = build_sampler(dataset=dataset_train, shuffle=True)
        dataloader_train = build_dataloader(dataset=dataset_train, sampler=sampler_train,
                                            batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"])

        if epoch >= config["ONLY_TRAIN_QUERY_UPDATER_AFTER"]:
            optimizer.param_groups[0]["lr"] = 0.0
            optimizer.param_groups[1]["lr"] = 0.0
            optimizer.param_groups[3]["lr"] = 0.0
        lrs = [optimizer.param_groups[_]["lr"] for _ in range(len(optimizer.param_groups))]
        assert len(lrs) == len(lr_names)
        lr_info = [{name: lr} for name, lr in zip(lr_names, lrs)]
        train_logger.show(head=f"[Epoch {epoch}] lr={lr_info}")
        train_logger.write(head=f"[Epoch {epoch}] lr={lr_info}")
        default_lr_idx = -1
        for _ in range(len(lr_names)):
            if lr_names[_] == "lr":
                default_lr_idx = _
        train_logger.tb_add_scalar(tag="lr", scalar_value=lrs[default_lr_idx], global_step=epoch, mode="epochs")

        no_grad_frames = None
        if "NO_GRAD_FRAMES" in config:
            for i in range(len(config["NO_GRAD_STEPS"])):
                if epoch >= config["NO_GRAD_STEPS"][i]:
                    no_grad_frames = config["NO_GRAD_FRAMES"][i]
                    break

        train_one_epoch(
            model=model,
            train_states=train_states,
            max_norm=config["CLIP_MAX_NORM"],
            dataloader=dataloader_train,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            logger=train_logger,
            accumulation_steps=config["ACCUMULATION_STEPS"],
            use_dab=config["USE_DAB"],
            multi_checkpoint=multi_checkpoint,
            no_grad_frames=no_grad_frames
        )
        scheduler.step()
        train_states["start_epoch"] += 1
        if multi_checkpoint is True:
            pass
        else:
            if config["DATASET"] == "DanceTrack" or config["EPOCHS"] < 100 or (epoch + 1) % 5 == 0:
                save_path = os.path.join(config["OUTPUTS_DIR"], f"checkpoint_{epoch}.pth")
                link_path = os.path.join(config["OUTPUTS_DIR"], f"last_checkpoint.pth")
                save_checkpoint(
                    model=model,
                    path=save_path,
                    states=train_states,
                    optimizer=optimizer,
                    scheduler=scheduler
                )
                link_checkpoint(save_path, link_path)

        if (epoch + 1) % config["EVAL_EPOCHS"] == 0:
            eval_model(config, model, outputs_dir, val_split, writer=tb_writer, epoch=epoch)

    return


def eval_model(config: dict, model: SambaMOTR, outputs_dir: str, val_split: str, writer: tb.SummaryWriter, epoch: int):
    # Submit
    data_root = config["DATA_ROOT"]
    dataset_name = config["DATASET"]
    use_dab = config["USE_DAB"]
    det_score_thresh = config["DET_SCORE_THRESH"]
    track_score_thresh = config["TRACK_SCORE_THRESH"]
    result_score_thresh = config["RESULT_SCORE_THRESH"]
    use_motion = config["USE_MOTION"]
    motion_min_length = config["MOTION_MIN_LENGTH"]
    motion_max_length = config["MOTION_MAX_LENGTH"]
    motion_lambda = config["MOTION_LAMBDA"]
    miss_tolerance = config["MISS_TOLERANCE"]
    interval=config["EVAL_INTERVAL"] if "EVAL_INTERVAL" in config else 1

    data_dir = os.path.join(data_root, dataset_name)
    if dataset_name == "DanceTrack" or dataset_name == "SportsMOT":
        data_split_dir = os.path.join(data_dir, val_split)
    elif dataset_name == "BFT":
        data_split_dir = os.path.join(data_dir, val_split)
        gt_split_dir = os.path.join(data_dir, "annotations_mot", val_split)
    elif dataset_name == "BDD100K":
        data_split_dir = os.path.join(data_dir, "images/track/", val_split)
    elif "MOT17" in dataset_name:
        benchmark = "MOT17"
        if "mot15" in val_split:
            data_dir = os.path.join(data_root, "MOT15")
            val_split = "train"
            benchmark = "MOT15"
            data_split_dir = os.path.join(data_dir, "images", "train")
        else:
            data_split_dir = os.path.join(data_dir, "images", val_split)
    else:
        raise NotImplementedError(f"Eval DOES NOT support dataset '{dataset_name}'")

    seq_names = os.listdir(data_split_dir)

    if is_distributed():
        total_seq_names = seq_names
        seq_names = []
        for i in range(len(total_seq_names)):
            if i % distributed_world_size() == distributed_rank():
                seq_names.append(total_seq_names[i])

    for seq_name in seq_names:
        seq_name = str(seq_name)
        submitter = Submitter(
            dataset_name=dataset_name,
            split_dir=data_split_dir,
            seq_name=seq_name,
            outputs_dir=outputs_dir,
            model=model,
            use_dab=use_dab,
            det_score_thresh=det_score_thresh,
            track_score_thresh=track_score_thresh,
            result_score_thresh=result_score_thresh,
            use_motion=use_motion,
            motion_min_length=motion_min_length,
            motion_max_length=motion_max_length,
            motion_lambda=motion_lambda,
            miss_tolerance=miss_tolerance,
            progress_bar=False,
            interval=interval
        )
        submitter.run()

    if is_distributed():
        dist.barrier()

    # Eval
    if is_main_process():
        tracker_dir = os.path.join(outputs_dir, "tracker")
        tracker_mv_dir = os.path.join(outputs_dir, f"checkpoint_{epoch}" + "_tracker")

        if os.path.exists(tracker_mv_dir):
            import shutil
            shutil.rmtree(tracker_mv_dir)
        os.system(f"mv {tracker_dir} {tracker_mv_dir}")
        
        if dataset_name == "DanceTrack" or dataset_name == "SportsMOT":
            os.system(f"python3 SparseTrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {val_split}  "
                    f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {data_split_dir} "
                    f"--SEQMAP_FILE {os.path.join(data_dir, f'{val_split}_seqmap.txt')} "
                    f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                    f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                    f"--TRACKERS_FOLDER {tracker_mv_dir} --EVAL_INTERVAL {interval}")
        elif dataset_name == "BFT":
            os.system(f"python3 SparseTrackEval/scripts/run_bft.py --SPLIT_TO_EVAL {val_split}  "
                    f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_split_dir} "
                    f"--GT_LOC_FORMAT {{gt_folder}}/{{seq}}.txt "
                    f"--SEQMAP_FILE {os.path.join(data_dir, f'{val_split}_seqmap.txt')} "
                    f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                    f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                    f"--TRACKERS_FOLDER {tracker_mv_dir} --EVAL_INTERVAL {interval}")
        elif "MOT17" in dataset_name:
            os.system(f"python3 SparseTrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {val_split}  "
                    f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {data_split_dir} "
                    f"--SEQMAP_FILE {os.path.join(data_dir, f'{val_split}_seqmap.txt')} "
                    f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                    f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                    f"--TRACKERS_FOLDER {tracker_mv_dir} --BENCHMARK {benchmark} --EVAL_INTERVAL {interval}")
        else:
            raise NotImplementedError(f"Do not support this Dataset name: {dataset_name}")

        if dataset_name == "BFT":
            metric_path = os.path.join(tracker_mv_dir, "bird_summary.txt")
        else:
            metric_path = os.path.join(tracker_mv_dir, "pedestrian_summary.txt")
        with open(metric_path) as f:
            metric_names = f.readline()[:-1].split(" ")
            metric_values = f.readline()[:-1].split(" ")
        metrics = {
            n: float(v) for n, v in zip(metric_names, metric_values)
        }
        metrics_to_tensorboard(writer=writer, metrics=metrics, epoch=epoch)

    return


def train_one_epoch(model: SambaMOTR, train_states: dict, max_norm: float,
                    dataloader: DataLoader, criterion: ClipCriterion, optimizer: torch.optim,
                    epoch: int, logger: Logger,
                    accumulation_steps: int = 1, use_dab: bool = False,
                    multi_checkpoint: bool = False,
                    no_grad_frames: int | None = None):
    """
    Args:
        model: Model.
        train_states:
        max_norm: clip max norm.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Training optimizer.
        epoch: Current epoch.
        logger: unified logger.
        accumulation_steps:
        use_dab:
        multi_checkpoint:
        no_grad_frames:

    Returns:
        Logs
    """
    model.train()
    optimizer.zero_grad()
    device = next(get_model(model).parameters()).device

    dataloader_len = len(dataloader)
    metric_log = MetricLog()
    epoch_start_timestamp = time.time()
    for i, batch in enumerate(dataloader):
        iter_start_timestamp = time.time()
        tracks = TrackInstances.init_tracks(batch=batch,
                                            hidden_dim=get_model(model).hidden_dim,
                                            num_classes=get_model(model).num_classes,
                                            state_dim=getattr(get_model(model).query_updater, "state_dim", 0),
                                            expand=getattr(get_model(model).query_updater, "expand", 0),
                                            num_layers=getattr(get_model(model).query_updater, "num_layers", 0),
                                            conv_dim=getattr(get_model(model).query_updater, "conv_dim", 0),
                                            device=device, use_dab=use_dab)
        criterion.init_a_clip(batch=batch,
                              hidden_dim=get_model(model).hidden_dim,
                              num_classes=get_model(model).num_classes,
                              state_dim=getattr(get_model(model).query_updater, "state_dim", 0),
                              expand=getattr(get_model(model).query_updater, "expand", 0),
                              num_layers=getattr(get_model(model).query_updater, "num_layers", 0),
                              conv_dim=getattr(get_model(model).query_updater, "conv_dim", 0),
                              device=device)

        for frame_idx in range(len(batch["imgs"][0])):
            if no_grad_frames is None or frame_idx >= no_grad_frames:
                frame = [fs[frame_idx] for fs in batch["imgs"]]
                for f in frame:
                    f.requires_grad_(False)
                frame = tensor_list_to_nested_tensor(tensor_list=frame).to(device)
                res = model(frame=frame, tracks=tracks)
                previous_tracks, new_tracks, unmatched_dets = criterion.process_single_frame(
                    model_outputs=res,
                    tracked_instances=tracks,
                    frame_idx=frame_idx
                )
                if frame_idx < len(batch["imgs"][0]) - 1:
                    tracks = get_model(model).postprocess_single_frame(
                        previous_tracks, new_tracks, unmatched_dets, intervals=batch["intervals"])
            else:
                with torch.no_grad():
                    frame = [fs[frame_idx] for fs in batch["imgs"]]
                    for f in frame:
                        f.requires_grad_(False)
                    frame = tensor_list_to_nested_tensor(tensor_list=frame).to(device)
                    res = model(frame=frame, tracks=tracks)
                    previous_tracks, new_tracks, unmatched_dets = criterion.process_single_frame(
                        model_outputs=res,
                        tracked_instances=tracks,
                        frame_idx=frame_idx
                    )
                    if frame_idx < len(batch["imgs"][0]) - 1:
                        tracks = get_model(model).postprocess_single_frame(
                            previous_tracks, new_tracks, unmatched_dets, intervals=batch["intervals"], no_augment=frame_idx <= no_grad_frames-1)

        loss_dict, log_dict = criterion.get_mean_by_n_gts()
        loss = criterion.get_sum_loss_dict(loss_dict=loss_dict)

        # Metrics log
        metric_log.update(name="total_loss", value=loss.item())
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            else:
                pass
            optimizer.step()
            optimizer.zero_grad()

        # For logging
        for log_k in log_dict:
            metric_log.update(name=log_k, value=log_dict[log_k][0])
        iter_end_timestamp = time.time()
        metric_log.update(name="time per iter", value=iter_end_timestamp-iter_start_timestamp)
        # Outputs logs
        if i % 100 == 0:
            metric_log.sync()
            # max_memory = max([torch.cuda.max_memory_allocated(torch.device('cuda', i))
            #                   for i in range(distributed_world_size())]) // (1024**2)
            
            # Calculate max memory per GPU in local node
            max_memory_local = max([
                torch.cuda.max_memory_allocated(torch.device('cuda', i))
                for i in range(torch.cuda.device_count())
            ]) // (1024**2)

            # Reduce the maximum across all nodes
            max_memory_global = torch.tensor(max_memory_local).to(torch.device('cuda'))
            if is_distributed():
                dist.reduce(max_memory_global, dst=0, op=dist.ReduceOp.MAX)

            second_per_iter = metric_log.metrics["time per iter"].avg
            logger.show(head=f"[Epoch={epoch}, Iter={i}, "
                             f"{second_per_iter:.2f}s/iter, "
                             f"{i}/{dataloader_len} iters, "
                             f"rest time: {int(second_per_iter * (dataloader_len - i) // 60)} min, "
                             f"Max Memory={max_memory_global}MB]",
                        log=metric_log)
            logger.write(head=f"[Epoch={epoch}, Iter={i}/{dataloader_len}]",
                         log=metric_log, filename="log.txt", mode="a")
            logger.tb_add_metric_log(log=metric_log, steps=train_states["global_iters"], mode="iters")

        if multi_checkpoint:
            if i % 100 == 0 and is_main_process():
                save_checkpoint(
                    model=model,
                    path=os.path.join(logger.logdir[:-5], f"checkpoint_{int(i // 100)}.pth")
                )

        train_states["global_iters"] += 1

    # Epoch end
    metric_log.sync()
    epoch_end_timestamp = time.time()
    epoch_minutes = int((epoch_end_timestamp - epoch_start_timestamp) // 60)
    logger.show(head=f"[Epoch: {epoch}, Total Time: {epoch_minutes}min]",
                log=metric_log)
    logger.write(head=f"[Epoch: {epoch}, Total Time: {epoch_minutes}min]",
                 log=metric_log, filename="log.txt", mode="a")
    logger.tb_add_metric_log(log=metric_log, steps=epoch, mode="epochs")

    return


def get_param_groups(config: dict, model: nn.Module) -> Tuple[List[Dict], List[str]]:
    def match_keywords(name: str, keywords: List[str]):
        matched = False
        for keyword in keywords:
            if keyword in name:
                matched = True
                break
        return matched
    # keywords
    backbone_keywords = ["backbone.backbone"]
    points_keywords = ["reference_points", "sampling_offsets"]
    query_updater_keywords = ["query_updater"]
    param_groups = [
        {   # backbone
            "params": [p for n, p in model.named_parameters() if match_keywords(n, backbone_keywords) and p.requires_grad],
            "lr": config["LR_BACKBONE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if match_keywords(n, points_keywords)
                       and p.requires_grad],
            "lr": config["LR_POINTS"]
        },
        {
            "params": [p for n, p in model.named_parameters() if match_keywords(n, query_updater_keywords)
                       and p.requires_grad],
            "lr": config["LR"]
        },
        {
            "params": [p for n, p in model.named_parameters() if not match_keywords(n, backbone_keywords)
                       and not match_keywords(n, points_keywords)
                       and not match_keywords(n, query_updater_keywords)
                       and p.requires_grad],
            "lr": config["LR"]
        }
    ]
    return param_groups, ["lr_backbone", "lr_points", "lr_query_updater", "lr"]
