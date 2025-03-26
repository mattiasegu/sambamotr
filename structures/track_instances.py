import torch

from typing import List


class TrackInstances:
    """
    Tracked Instances.
    """
    def __init__(self, frame_height: float = 1.0, frame_width: float = 1.0,
                 hidden_dim: int = 256, num_classes: int = 1, state_dim: int = 8,
                 expand: int = 2, num_layers: int = 2, conv_dim: int = 4,
                 use_dab: bool = False):
        self.use_dab = use_dab
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.expand = expand
        self.num_layers = num_layers
        self.conv_dim = conv_dim
        self.num_classes = num_classes
        if self.use_dab:
            self.ref_pts = torch.zeros((0, 4))
            self.query_embed = torch.zeros((0, hidden_dim))
        else:
            # self.ref_pts = torch.zeros((0, 2))
            self.ref_pts = torch.zeros((0, 4))
            self.query_embed = torch.zeros((0, 2 * hidden_dim))
        self.ids = torch.zeros((0,), dtype=torch.long)
        self.boxes = torch.zeros((0, 4))
        self.labels = torch.zeros((0,), dtype=torch.long)
        self.logits = torch.zeros((0, self.num_classes))
        self.matched_idx = torch.zeros((0, ), dtype=torch.long)
        self.output_embed = torch.zeros((0, self.hidden_dim))
        self.disappear_time = torch.zeros((0,), dtype=torch.long)
        self.scores = torch.zeros((0,), dtype=torch.float)
        self.area = torch.zeros((0,), dtype=torch.float)
        self.iou = torch.zeros((0,), dtype=torch.float)
        self.last_output = torch.zeros((0, self.hidden_dim), dtype=torch.float)
        self.long_memory = torch.zeros((0, self.hidden_dim), dtype=torch.float)
        self.last_appear_boxes = torch.zeros((0, 4))

        # Samba
        self.hidden_state = torch.zeros((0, num_layers, hidden_dim * expand, state_dim), dtype=torch.float)
        self.conv_history = torch.zeros((0, num_layers, conv_dim, hidden_dim * expand), dtype=torch.float)

    def to(self, device):
        res = TrackInstances(frame_height=self.frame_height, frame_width=self.frame_width,
                             hidden_dim=self.hidden_dim, num_classes=self.num_classes,
                             state_dim=self.state_dim, expand=self.expand,
                             num_layers=self.num_layers,  conv_dim=self.conv_dim)
        for k, v in vars(self).items():
            if hasattr(v, "to"):
                v = v.to(device)
            res.__setattr__(k, v)
        return res

    def __len__(self):
        assert self.ref_pts.shape[0] == self.query_embed.shape[0]
        return max(self.query_embed.shape[0], self.labels.shape[0])

    def __getitem__(self, item: int | slice | torch.BoolTensor) -> "TrackInstances":
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("TrackInstances index out of range!")
            else:
                item = slice(item, None, len(self))
        res = TrackInstances(frame_height=self.frame_height, frame_width=self.frame_width,
                             hidden_dim=self.hidden_dim, num_classes=self.num_classes,
                             state_dim=self.state_dim, expand=self.expand,
                             num_layers=self.num_layers, conv_dim=self.conv_dim)
        for k, v in vars(self).items():
            if hasattr(v, "__getitem__") and v.shape[0] != 0:
                try:
                    res.__setattr__(k, v[item])
                except:
                    print(f"Error caught for attribute {k}")
            else:
                res.__setattr__(k, v)
        return res

    @staticmethod
    def init_tracks(batch: dict, hidden_dim: int, num_classes: int,
                    state_dim: int, expand: int, num_layers: int, conv_dim: int,
                    device="cpu", use_dab: bool = False):
        """
        Init tracks for a batch.
        """
        tracks_list = []
        h_max, w_max = 0, 0
        for i in range(len(batch["imgs"])):
            h_max = max(batch["imgs"][i][0].shape[-2], h_max)
            w_max = max(batch["imgs"][i][0].shape[-1], w_max)
        for i in range(len(batch["imgs"])):
            tracks_list.append(TrackInstances(
                frame_height=float(batch["imgs"][i][0].shape[-2] / h_max),
                frame_width=float(batch["imgs"][i][0].shape[-1] / w_max),
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                state_dim=state_dim,
                expand=expand,
                num_layers=num_layers,
                conv_dim=conv_dim,
                use_dab=use_dab,
            ).to(device))
        return tracks_list

    @staticmethod
    def cat_tracked_instances(tracked1: "TrackInstances", tracked2: "TrackInstances"):
        res = TrackInstances(frame_height=tracked1.frame_height, frame_width=tracked1.frame_width,
                             hidden_dim=tracked1.hidden_dim, state_dim=tracked1.state_dim, expand=tracked1.expand, 
                             num_layers=tracked1.num_layers, conv_dim=tracked1.conv_dim)

        for k, v in vars(tracked1).items():
            if type(v) is torch.Tensor:
                res.__setattr__(k, torch.cat((getattr(tracked1, k), getattr(tracked2, k))))
        return res

    @staticmethod
    def tracks_to_meta_tensors(tracks: List["TrackInstances"]):
        keys = [k for k in vars(tracks[0]) if type(getattr(tracks[0], k)) is torch.Tensor]
        meta = {
            "frame_height": [],
            "frame_width": [],
            "hidden_dim": [],
            "num_classes": [],
            "state_dim": [],
            "expand": [],
            "num_layers": [],
            "conv_dim": [],
            "keys": keys,
        }
        tensors = []
        for b in range(len(tracks)):
            meta["frame_height"].append(tracks[b].frame_height)
            meta["frame_width"].append(tracks[b].frame_width)
            meta["hidden_dim"].append(tracks[b].hidden_dim)
            meta["num_classes"].append(tracks[b].num_classes)
            meta["state_dim"].append(tracks[b].state_dim)
            meta["expand"].append(tracks[b].expand)
            meta["num_layers"].append(tracks[b].num_layers)
            meta["conv_dim"].append(tracks[b].conv_dim)
            for k in keys:
                tensors.append(getattr(tracks[b], k))
        return meta, tensors

    @staticmethod
    def meta_tensors_to_tracks(meta: dict, tensors: List[torch.Tensor]):
        tracks = []
        for b in range(len(meta["frame_height"])):
            track = TrackInstances(
                frame_height=meta["frame_height"][b],
                frame_width=meta["frame_width"][b],
                hidden_dim=meta["hidden_dim"][b],
                num_classes=meta["num_classes"][b],
                state_dim=meta["state_dim"][b],
                expand=meta["expand"][b],
                num_layers=meta["num_layers"][b],
                conv_dim=meta["conv_dim"][b]
            )
            for i, k in enumerate(meta["keys"]):
                setattr(track, k, tensors[i + b * len(meta["keys"])])
            tracks.append(track)
        return tracks
