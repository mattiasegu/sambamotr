import os
import math
import torch
import torch.nn as nn

from typing import List
from .utils import pos_to_pos_embed, logits_to_scores
from torch.utils.checkpoint import checkpoint

from .ffn import FFN
from .mlp import MLP
from structures.track_instances import TrackInstances
from utils.utils import inverse_sigmoid
from utils.box_ops import box_cxcywh_to_xyxy, box_iou_union

from .query_updaters.samba import Samba


class BatchDropout(nn.Module):
    def __init__(self, p=0.5):
        """
        Initializes the BatchDropout layer.
        
        Args:
        p (float): Probability of dropping a sample in the batch.
        """
        super(BatchDropout, self).__init__()
        self.p = p  # Store the probability of dropping a sample

    def forward(self, x):
        """
        The forward pass for applying dropout to entire samples in the batch.

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, features).

        Returns:
        torch.Tensor: Output tensor after applying dropout.
        """
        if not self.training:
            return x  # Return the input directly if the model is in evaluation mode

        # Create a dropout mask with a probability `p` for dropping
        batch_size = x.size(0)
        mask = torch.rand(batch_size, device=x.device) > self.p  # This creates a boolean mask
        
        # Convert the boolean mask to the same dtype as `x`
        mask = mask.float()
        mask = mask.unsqueeze(1)  # Adjust mask shape to be broadcastable to the shape of `x`
        
        return x * mask  # Apply mask


class UnifiedQueryUpdater(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 ffn_dim: int,
                 num_heads: int,
                 state_dim: int,
                 expand: int,
                 num_layers: int,
                 num_attn_layers: int,
                 conv_dim: int,
                 with_self_attn: bool,
                 with_self_attn_prior: bool,
                 fps_invariant: bool,
                 tp_drop_ratio: float,
                 fp_insert_ratio: float,
                 dropout: float,
                 use_checkpoint: bool,
                 use_dab: bool,
                 update_threshold: float,
                 long_memory_lambda: float,
                 sequence_model: str = 'samba',
                 with_conv: bool = True,
                 inner_layernorms: bool = True,
                 with_input_layernorm: bool = False,
                 with_residual: bool = False,
                 with_residual_scale: bool = False,
                 visualize: bool = False,
                 with_masking: bool = False,
                 with_detach: bool = False,
                 with_ref_pts_residual: bool = False,
                 update_only_pos: bool = False):
        """
        Unified query updater supporting multiple sequence models (e.g., Samba, xLSTM).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.tp_drop_ratio = tp_drop_ratio
        self.fp_insert_ratio = fp_insert_ratio
        self.dropout = dropout
        self.observation_dropout = BatchDropout(self.dropout)

        # Samba
        self.num_heads = num_heads
        self.state_dim = state_dim
        self.expand = expand
        self.num_layers = num_layers
        self.num_attn_layers = num_attn_layers
        self.conv_dim = conv_dim
        self.with_self_attn = with_self_attn
        self.with_self_attn_prior = with_self_attn_prior
        self.fps_invariant = fps_invariant

        self.use_checkpoint = use_checkpoint
        self.use_dab = use_dab
        self.visualize = visualize
        self.with_residual = with_residual
        self.with_residual_scale = with_residual_scale

        # Sequence Model Initialization
        self.sequence_model_type = sequence_model.lower()

        if self.sequence_model_type == 'samba':
            self.sequence_model = Samba(num_layers=num_layers, d_model=hidden_dim,
                                        layer_cfg = dict(
                                            d_model=hidden_dim,
                                            d_state=state_dim,
                                            expand=expand,
                                            dt_rank='auto',
                                            d_conv=conv_dim,
                                            conv_bias=True,
                                            bias=False,
                                            with_self_attn=with_self_attn,
                                            with_self_attn_prior=with_self_attn_prior,
                                            num_attn_layers=num_attn_layers,
                                            self_attn_cfg=dict(
                                                embed_dims=hidden_dim, num_heads=self.num_heads, dropout=0.0),
                                            ffn_cfg=dict(
                                                embed_dims=hidden_dim,
                                                feedforward_channels=ffn_dim,
                                                num_fcs=2,
                                                ffn_drop=0.,
                                                act_cfg=dict(type='ReLU', inplace=True)),
                                             with_conv=with_conv,
                                             inner_layernorms=inner_layernorms,
                                             with_input_layernorm=with_input_layernorm))
        else:
            raise ValueError(f"Unsupported sequence model: {self.sequence_model_type}")

        self.update_threshold = update_threshold

        self.query_pos_head = MLP(
            input_dim=self.hidden_dim * 2,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=2
        )

        if with_residual:
            self.query_feat_ffn = FFN(d_model=self.hidden_dim, d_ffn=self.ffn_dim, dropout=0.0)
            self.norm_emb = nn.LayerNorm(self.hidden_dim)
            if with_residual_scale:
                self.residual_scale = nn.Parameter(torch.ones([]))
                self.residual_scale_pos = nn.Parameter(torch.ones([]))

        if not use_dab:
            self.linear_pos1 = nn.Linear(256, 256)
            self.linear_pos2 = nn.Linear(256, 256)
            self.norm_pos = nn.LayerNorm(256)
            self.activation = nn.ReLU(inplace=True)

        # Masking-specific attributes
        self.with_masking = with_masking
        self.with_detach = with_detach
        self.with_ref_pts_residual = with_ref_pts_residual
        self.update_only_pos = update_only_pos

        if self.with_ref_pts_residual:
            self.bbox_embed = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=4, num_layers=3)
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                previous_tracks: List[TrackInstances],
                new_tracks: List[TrackInstances],
                unmatched_dets: List[TrackInstances] | None,
                intervals=List[int],
                no_augment: bool = False):
        tracks = self.select_active_tracks(previous_tracks, new_tracks, unmatched_dets, no_augment=no_augment)
        tracks = self.update_tracks_embedding(tracks=tracks, intervals=intervals, no_augment=no_augment)
        return tracks

    def update_tracks_embedding(self, tracks: List[TrackInstances], intervals: List[int], no_augment: bool = False):
        for b in range(len(tracks)):
            scores = torch.max(logits_to_scores(logits=tracks[b].logits), dim=1).values
            is_pos = scores > self.update_threshold

            if self.visualize:
                self.visualize_tracks(tracks[b], scores, prefix="current")

            if self.use_dab:
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b].boxes[is_pos].detach().clone())
            else:
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b].boxes[is_pos].detach().clone())

            output_pos = pos_to_pos_embed(tracks[b].ref_pts.sigmoid(), num_pos_feats=self.hidden_dim // 2)
            output_pos = self.query_pos_head(output_pos)
            output_embed = tracks[b].output_embed

            if self.with_masking:
                if not no_augment:
                    output_embed = self.observation_dropout(output_embed)
                output_embed[~is_pos] = 0.0 * output_embed[~is_pos]
                output_pos[~is_pos] = 0.0 * output_pos[~is_pos]

            rate = 1 / intervals[b] if self.fps_invariant else 1
            hidden_state = tracks[b].hidden_state
            conv_history = tracks[b].conv_history

            if self.sequence_model_type == 'samba':
                output_embed, hidden_state, conv_history = self.sequence_model(
                    output_embed.unsqueeze(0),
                    hidden_state.unsqueeze(0),
                    conv_history.unsqueeze(0),
                    output_pos.unsqueeze(0),
                    rate=rate)
                hidden_state = hidden_state.squeeze(0)
                conv_history = conv_history.squeeze(0)

            tracks[b].hidden_state = hidden_state
            tracks[b].conv_history = conv_history
            output_embed = output_embed.squeeze(0)

            # unlike MeMOTR, we update the embed also for low-confidence boxes since Samba takes care of masked observations from context
            update = is_pos if self.update_only_pos else scores >= 0.0
            if self.use_dab:
                if self.with_residual:
                    new_query_embed = self.query_feat_ffn(output_embed)
                    query_embed = tracks[b].query_embed
                    if self.with_detach:
                        query_embed = query_embed.detach().clone()
                    if self.with_residual_scale:
                        query_embed = query_embed + self.residual_scale.exp() * new_query_embed
                    else:
                        query_embed = query_embed + new_query_embed
                    query_embed = self.norm_emb(query_embed)
                    tracks[b].query_embed[update] = query_embed[update]
                else:
                    tracks[b].query_embed[update] = output_embed[update]
            else:
                if self.with_residual:
                    new_query_embed = self.query_feat_ffn(output_embed)
                    query_embed = tracks[b].query_embed[:, self.hidden_dim:]
                    if self.with_detach:
                        query_embed = query_embed.detach().clone()
                    if self.with_residual_scale:
                        query_embed = query_embed + self.residual_scale.exp() * new_query_embed
                    else:
                        query_embed = query_embed + new_query_embed
                    query_embed = self.norm_emb(query_embed)
                    tracks[b].query_embed[:, self.hidden_dim:][update] = query_embed[update]
                else:
                    tracks[b].query_embed[:, self.hidden_dim:][update] = output_embed[update]

                # Update query pos, which is not used in DAB-D-DETR framework:
                new_query_pos = self.linear_pos2(self.activation(self.linear_pos1(output_embed)))
                query_pos = tracks[b].query_embed[:, :self.hidden_dim]
                if self.with_detach:
                    query_pos = query_pos.detach().clone()
                if self.with_residual_scale:
                    query_pos = query_pos + self.residual_scale_pos.exp() * new_query_pos
                else:
                    query_pos = query_pos + new_query_pos
                query_pos = self.norm_pos(query_pos)
                tracks[b].query_embed[:, :self.hidden_dim][update] = query_pos[update]

            if self.with_ref_pts_residual:
                tmp = self.bbox_embed(output_embed)
                if tracks[b].ref_pts.shape[-1] == 4:
                    tracks[b].ref_pts[update] = tmp[update] + tracks[b].ref_pts[update]
                else:
                    assert tracks[b].ref_pts.shape[-1] == 2
                    tracks[b].ref_pts[update] = tmp[..., :2][update] + tracks[b].ref_pts[update]

            if self.visualize:
                self.visualize_tracks(tracks[b], scores, prefix="next")

        return tracks

    def visualize_tracks(self, track, scores, prefix):
        os.makedirs("./outputs/visualize_tmp/query_updater/", exist_ok=True)
        torch.save(track.ref_pts.cpu(), f"./outputs/visualize_tmp/query_updater/{prefix}_ref_pts.tensor")
        torch.save(track.query_embed.cpu(), f"./outputs/visualize_tmp/query_updater/{prefix}_query_feat.tensor")
        torch.save(track.ids.cpu(), f"./outputs/visualize_tmp/query_updater/{prefix}_ids.tensor")
        torch.save(track.labels.cpu(), f"./outputs/visualize_tmp/query_updater/{prefix}_labels.tensor")
        torch.save(scores.cpu(), f"./outputs/visualize_tmp/query_updater/{prefix}_scores.tensor")

    def select_active_tracks(self, previous_tracks: List[TrackInstances],
                             new_tracks: List[TrackInstances],
                             unmatched_dets: List[TrackInstances],
                             no_augment: bool = False):
        tracks = []
        if self.training:
            for b in range(len(new_tracks)):
                # Update fields
                new_tracks[b].last_output = new_tracks[b].output_embed
                if self.use_dab:
                    new_tracks[b].long_memory = new_tracks[b].query_embed
                else:
                    new_tracks[b].long_memory = new_tracks[b].query_embed[:, self.hidden_dim:]
                unmatched_dets[b].last_output = unmatched_dets[b].output_embed
                if self.use_dab:
                    unmatched_dets[b].long_memory = unmatched_dets[b].query_embed
                else:
                    unmatched_dets[b].long_memory = unmatched_dets[b].query_embed[:, self.hidden_dim:]
                if self.tp_drop_ratio == 0.0 and self.fp_insert_ratio == 0.0:
                    active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[b], new_tracks[b])
                    active_tracks = TrackInstances.cat_tracked_instances(active_tracks, unmatched_dets[b])
                    scores = torch.max(logits_to_scores(logits=active_tracks.logits), dim=1).values
                    keep_idxes = (scores > self.update_threshold) | (active_tracks.ids >= 0)
                    active_tracks = active_tracks[keep_idxes]
                    active_tracks.ids[active_tracks.iou < 0.5] = -1
                else:
                    active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[b], new_tracks[b])
                    active_tracks = active_tracks[(active_tracks.iou > 0.5) & (active_tracks.ids >= 0)]
                    if self.tp_drop_ratio > 0.0 and not no_augment:
                        if len(active_tracks) > 0:
                            tp_keep_idx = torch.rand((len(active_tracks), )) > self.tp_drop_ratio
                            active_tracks = active_tracks[tp_keep_idx]
                    if self.fp_insert_ratio > 0.0 and not no_augment:
                        selected_active_tracks = active_tracks[
                            torch.bernoulli(
                                torch.ones((len(active_tracks), )) * self.fp_insert_ratio
                            ).bool()
                        ]
                        if len(unmatched_dets[b]) > 0 and len(selected_active_tracks) > 0:
                            fp_num = len(selected_active_tracks)
                            if fp_num >= len(unmatched_dets[b]):
                                insert_fp = unmatched_dets[b]
                            else:
                                selected_active_boxes = box_cxcywh_to_xyxy(selected_active_tracks.boxes)
                                unmatched_boxes = box_cxcywh_to_xyxy(unmatched_dets[b].boxes)
                                iou, _ = box_iou_union(unmatched_boxes, selected_active_boxes)
                                fp_idx = torch.max(iou, dim=0).indices
                                fp_idx = torch.unique(fp_idx)
                                insert_fp = unmatched_dets[b][fp_idx]
                            active_tracks = TrackInstances.cat_tracked_instances(active_tracks, insert_fp)

                if len(active_tracks) == 0:
                    device = self.query_pos_head.layers[0].weight.device
                    fake_tracks = TrackInstances(frame_height=1.0, frame_width=1.0, hidden_dim=self.hidden_dim,
                                                 state_dim=self.state_dim, expand=self.expand,
                                                 num_layers=self.num_layers, conv_dim=self.conv_dim
                                                 ).to(device=device)
                    if self.use_dab:
                        fake_tracks.query_embed = torch.randn((1, self.hidden_dim), dtype=torch.float,
                                                              device=device)
                    else:
                        fake_tracks.query_embed = torch.randn((1, 2 * self.hidden_dim), dtype=torch.float, device=device)
                    fake_tracks.output_embed = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
                    if self.use_dab:
                        fake_tracks.ref_pts = torch.randn((1, 4), dtype=torch.float, device=device)
                    else:
                        # fake_tracks.ref_pts = torch.randn((1, 2), dtype=torch.float, device=device)
                        fake_tracks.ref_pts = torch.randn((1, 4), dtype=torch.float, device=device)
                    fake_tracks.ids = torch.as_tensor([-2], dtype=torch.long, device=device)
                    fake_tracks.matched_idx = torch.as_tensor([-2], dtype=torch.long, device=device)
                    fake_tracks.boxes = torch.randn((1, 4), dtype=torch.float, device=device)
                    fake_tracks.logits = torch.randn((1, active_tracks.logits.shape[1]), dtype=torch.float, device=device)
                    fake_tracks.iou = torch.zeros((1,), dtype=torch.float, device=device)
                    fake_tracks.last_output = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
                    fake_tracks.long_memory = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
                    # Samba
                    fake_tracks.hidden_state = torch.zeros((1, self.num_layers, self.hidden_dim * self.expand, self.state_dim), dtype=torch.float, device=device)
                    fake_tracks.conv_history = torch.zeros((1, self.num_layers, self.conv_dim, self.hidden_dim * self.expand), dtype=torch.float, device=device)

                    active_tracks = fake_tracks
                tracks.append(active_tracks)
        else:
            # Eval only has B=1.
            assert len(previous_tracks) == 1 and len(new_tracks) == 1
            new_tracks[0].last_output = new_tracks[0].output_embed
            # new_tracks[0].long_memory = new_tracks[0].query_embed
            if self.use_dab:
                new_tracks[0].long_memory = new_tracks[0].query_embed
            else:
                new_tracks[0].long_memory = new_tracks[0].query_embed[:, self.hidden_dim:]
            active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[0], new_tracks[0])
            active_tracks = active_tracks[active_tracks.ids >= 0]
            tracks.append(active_tracks)
        return tracks


def build(config: dict):
    if config["QUERY_UPDATER"] == "UnifiedQueryUpdater":
        return UnifiedQueryUpdater(
                hidden_dim=config["HIDDEN_DIM"],
                ffn_dim=config["SAMBA_FFN_DIM"],
                num_heads=config["SAMBA_NUM_HEADS"],
                state_dim=config["STATE_DIM"],
                expand=config["EXPAND"],
                num_layers=config["SAMBA_NUM_LAYERS"],
                num_attn_layers=config["SAMBA_NUM_ATTN_LAYERS"],
                conv_dim=config["CONV_DIM"],
                with_self_attn=config["WITH_SELF_ATTN"],
                with_self_attn_prior=config["WITH_SELF_ATTN_PRIOR"] if "WITH_SELF_ATTN_PRIOR" in config else False,
                fps_invariant=config["FPS_INVARIANT"],
                dropout=config["SAMBA_DROPOUT"] if "SAMBA_DROPOUT" in config else 0.0,
                tp_drop_ratio=config["TP_DROP_RATE"] if "TP_DROP_RATE" in config else 0.0,
                fp_insert_ratio=config["FP_INSERT_RATE"] if "FP_INSERT_RATE" in config else 0.0,
                use_checkpoint=config["USE_CHECKPOINT"],
                use_dab=config["USE_DAB"],
                update_threshold=config["UPDATE_THRESH"],
                long_memory_lambda=config["LONG_MEMORY_LAMBDA"],
                visualize=config["VISUALIZE"],
                sequence_model=config["SEQUENCE_MODEL"],
                with_conv=config["SAMBA_WITH_CONV"] if "SAMBA_WITH_CONV" in config else True,
                inner_layernorms=config["SAMBA_WITH_INNER_LAYERNORMS"] if "SAMBA_WITH_INNER_LAYERNORMS" in config else False,
                with_input_layernorm=config["SAMBA_WITH_INPUT_LAYERNORM"] if "SAMBA_WITH_INPUT_LAYERNORM" in config else False,
                with_masking=config["MASKING"],
                with_residual=config["RESIDUAL"],
                with_residual_scale=config["RESIDUAL_SCALE"] if "RESIDUAL_SCALE" in config else False,
                with_detach=config["DETACH"] if "DETACH" in config else False,
                with_ref_pts_residual=config["REF_PTS_RESIDUAL"],
                update_only_pos=config["UPDATE_ONLY_POS"],
            )
    else:
        ValueError(f"Do not support query updater '{config['QUERY_UPDATER']}'")

