import torch

from utils.utils import distributed_rank
from .sambamotr import build as build_sambamotr


def build_model(config: dict):
    
    model = build_sambamotr(config=config)
    # if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
    #     model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    # else:
    #     model.to(device=torch.device(config["DEVICE"]))
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.cuda.current_device())
    else:
        model.to(device=torch.device(config["DEVICE"]))
    return model
