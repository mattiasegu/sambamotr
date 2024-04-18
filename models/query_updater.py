# Copyright (c) Ruopeng Gao. All Rights Reserved.
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

from .samba import Samba


class QueryUpdater(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int,
                 num_heads: int,
                 state_dim: int,
                 expand: int,
                 num_layers: int,
                 conv_dim: int,
                 with_self_attn: bool,
                 tp_drop_ratio: float, fp_insert_ratio: float,
                 dropout: float,
                 use_checkpoint: bool, use_dab: bool,
                 update_threshold: float, long_memory_lambda: float,
                 visualize: bool = False):
        super(QueryUpdater, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.tp_drop_ratio = tp_drop_ratio
        self.fp_insert_ratio = fp_insert_ratio
        self.dropout = dropout

        self.use_checkpoint = use_checkpoint
        self.use_dab = use_dab
        self.visualize = visualize

        self.update_threshold = update_threshold
        self.long_memory_lambda = long_memory_lambda

        # Samba
        self.num_heads = num_heads
        self.state_dim = state_dim
        self.expand = expand
        self.num_layers = num_layers
        self.conv_dim = conv_dim
        self.with_self_attn = with_self_attn

        self.confidence_weight_net = nn.Sequential(
            MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=self.hidden_dim, num_layers=2),
            nn.Sigmoid()
        )
        self.short_memory_fusion = MLP(input_dim=2*self.hidden_dim, hidden_dim=2*self.hidden_dim,
                                       output_dim=self.hidden_dim, num_layers=2)
        self.memory_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)
        self.memory_dropout = nn.Dropout(self.dropout)
        self.memory_norm = nn.LayerNorm(self.hidden_dim)
        self.memory_ffn = FFN(d_model=self.hidden_dim, d_ffn=self.ffn_dim, dropout=self.dropout)
        self.query_feat_dropout = nn.Dropout(self.dropout)
        self.query_feat_norm = nn.LayerNorm(self.hidden_dim)
        self.query_feat_ffn = FFN(d_model=self.hidden_dim, d_ffn=self.ffn_dim, dropout=self.dropout)
        self.query_pos_head = MLP(
            input_dim=self.hidden_dim*2,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=2
        )

        if self.use_dab is False:   # D-DETR, use this module to update the
            self.linear_pos1 = nn.Linear(256, 256)
            self.linear_pos2 = nn.Linear(256, 256)
            self.norm_pos = nn.LayerNorm(256)
            self.activation = nn.ReLU(inplace=True)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                previous_tracks: List[TrackInstances],
                new_tracks: List[TrackInstances],
                unmatched_dets: List[TrackInstances] | None,
                intervals = List[int],
                no_augment: bool = False):
        tracks = self.select_active_tracks(previous_tracks, new_tracks, unmatched_dets, no_augment=no_augment)
        tracks = self.update_tracks_embedding(tracks=tracks)

        return tracks

    def update_tracks_embedding(self, tracks: List[TrackInstances]):
        for b in range(len(tracks)):
            scores = torch.max(logits_to_scores(logits=tracks[b].logits), dim=1).values
            is_pos = scores > self.update_threshold
            if self.visualize:
                os.makedirs("./outputs/visualize_tmp/query_updater/", exist_ok=True)
                torch.save(tracks[b].ref_pts.cpu(), "./outputs/visualize_tmp/query_updater/current_ref_pts.tensor")
                # torch.save(tracks[b].query_embed[:, :].cpu(),
                #            "./outputs/visualize_tmp/query_updater/current_query_pos.tensor")
                # torch.save(tracks[b].query_embed[:, :].cpu(),
                #            "./outputs/visualize_tmp/query_updater/current_query_feat.tensor")
                torch.save(tracks[b].query_embed.cpu(),
                           "./outputs/visualize_tmp/query_updater/current_query_feat.tensor")
                torch.save(tracks[b].ids.cpu(), "./outputs/visualize_tmp/query_updater/current_ids.tensor")
                torch.save(tracks[b].labels.cpu(), "./outputs/visualize_tmp/query_updater/current_labels.tensor")
                torch.save(scores.cpu(), "./outputs/visualize_tmp/query_updater/current_scores.tensor")
            if self.use_dab:
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b][is_pos].boxes.detach().clone())
            else:
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b][is_pos].boxes.detach().clone())

            query_pos = pos_to_pos_embed(tracks[b].ref_pts.sigmoid(), num_pos_feats=self.hidden_dim//2)
            output_embed = tracks[b].output_embed
            last_output_embed = tracks[b].last_output
            long_memory = tracks[b].long_memory.detach()

            # Confidence Weight
            confidence_weight = self.confidence_weight_net(output_embed)

            # Adaptive Aggregation
            short_memory = self.short_memory_fusion(
                torch.cat((
                    confidence_weight * output_embed,
                    last_output_embed
                ), dim=-1)
            )

            # Query Feature Generate
            query_pos = self.query_pos_head(query_pos)
            q = short_memory + query_pos
            k = long_memory + query_pos
            tgt = output_embed
            # Attention
            tgt2 = self.memory_attn(q[None, :], k[None, :], tgt[None, :])[0][0, :]
            tgt = tgt + self.memory_dropout(tgt2)
            tgt = self.memory_norm(tgt)
            tgt = self.memory_ffn(tgt)
            # Long Memory ResNet
            query_feat = long_memory + self.query_feat_dropout(tgt)
            query_feat = self.query_feat_norm(query_feat)
            query_feat = self.query_feat_ffn(query_feat)

            # Update Long Memory
            long_memory = (1 - self.long_memory_lambda) * long_memory + \
                          self.long_memory_lambda * tracks[b].output_embed
            tracks[b].long_memory = tracks[b].long_memory * ~is_pos.reshape((is_pos.shape[0], 1)) + \
                                    long_memory * is_pos.reshape((is_pos.shape[0], 1))
            # Update Last Outputs Embedding
            tracks[b].last_output = tracks[b].last_output * ~is_pos.reshape((is_pos.shape[0], 1)) + \
                                    output_embed * is_pos.reshape((is_pos.shape[0], 1))

            if self.use_dab:
                tracks[b].query_embed[is_pos] = query_feat[is_pos]
            else:
                tracks[b].query_embed[:, self.hidden_dim:][is_pos] = query_feat[is_pos]
                # Update query pos, which is not appeared in DAB-D-DETR framework:
                new_query_pos = self.linear_pos2(self.activation(self.linear_pos1(output_embed)))
                query_pos = tracks[b].query_embed[:, :self.hidden_dim]
                query_pos = query_pos + new_query_pos
                query_pos = self.norm_pos(query_pos)
                tracks[b].query_embed[:, :self.hidden_dim][is_pos] = query_pos[is_pos]

            if self.visualize:
                torch.save(tracks[b].ref_pts.cpu(), "./outputs/visualize_tmp/query_updater/next_ref_pts.tensor")
                # torch.save(tracks[b].query_embed[:, :self.hidden_dim].cpu(),
                #            "./outputs/visualize_tmp/query_updater/next_query_pos.tensor")
                # torch.save(tracks[b].query_embed[:, self.hidden_dim:].cpu(),
                #            "./outputs/visualize_tmp/query_updater/next_query_feat.tensor")
                torch.save(tracks[b].query_embed.cpu(),
                           "./outputs/visualize_tmp/query_updater/next_query_feat.tensor")
                torch.save(tracks[b].ids.cpu(), "./outputs/visualize_tmp/query_updater/next_ids.tensor")
                torch.save(tracks[b].labels.cpu(), "./outputs/visualize_tmp/query_updater/next_labels.tensor")
                torch.save(scores.cpu(), "./outputs/visualize_tmp/query_updater/next_scores.tensor")

        return tracks

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
                    device = next(self.query_feat_ffn.parameters()).device
                    fake_tracks = TrackInstances(frame_height=1.0, frame_width=1.0, hidden_dim=self.hidden_dim,
                                                 state_dim=0, expand=0, num_layers=0, conv_dim=0).to(
                        device=device)
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
                    fake_tracks.hidden_state = torch.zeros((1, self.hidden_dim * self.expand, self.state_dim), dtype=torch.float, device=device)
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
    

class SambaQueryUpdater(nn.Module):
    def __init__(self,
                hidden_dim: int,
                ffn_dim: int,
                num_heads: int,
                state_dim: int,
                expand: int,
                num_layers: int,
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
                with_residual: bool = False,
                visualize: bool = False):
        """
        Init a query updater.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.tp_drop_ratio = tp_drop_ratio
        self.fp_insert_ratio = fp_insert_ratio
        self.dropout = dropout

        # Samba
        self.num_heads = num_heads
        self.state_dim = state_dim
        self.expand = expand
        self.num_layers = num_layers
        self.conv_dim = conv_dim
        self.with_self_attn = with_self_attn
        self.with_self_attn_prior = with_self_attn_prior
        self.fps_invariant = fps_invariant

        self.use_checkpoint = use_checkpoint
        self.use_dab = use_dab
        self.visualize = visualize
        self.with_residual = with_residual

        self.samba = Samba(num_layers=num_layers,
                            d_model=hidden_dim,
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
                                self_attn_cfg=dict(
                                    embed_dims=hidden_dim, num_heads=self.num_heads, dropout=0.0),
                                ffn_cfg=dict(
                                    embed_dims=hidden_dim,
                                    feedforward_channels=ffn_dim,
                                    num_fcs=2,
                                    ffn_drop=0.,
                                    act_cfg=dict(type='ReLU', inplace=True))))

        self.update_threshold = update_threshold

        self.query_pos_head = MLP(
            input_dim=self.hidden_dim*2,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=2
        )

        if self.with_residual:
            self.query_feat_ffn = FFN(d_model=self.hidden_dim, d_ffn=self.ffn_dim, dropout=self.dropout)
            self.norm_emb = nn.LayerNorm(self.hidden_dim)

        if self.use_dab is False:   # D-DETR, use this module to update the
            self.linear_pos1 = nn.Linear(256, 256)
            self.linear_pos2 = nn.Linear(256, 256)
            self.norm_pos = nn.LayerNorm(256)
            self.activation = nn.ReLU(inplace=True)

        self.reset_parameters()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                previous_tracks: List[TrackInstances],
                new_tracks: List[TrackInstances],
                unmatched_dets: List[TrackInstances] | None,
                intervals = List[int],
                no_augment: bool = False):
        tracks = self.select_active_tracks(previous_tracks, new_tracks, unmatched_dets, no_augment=no_augment)
        tracks = self.update_tracks_embedding(tracks=tracks, intervals=intervals)

        return tracks
    
    def update_tracks_embedding(self, tracks: List[TrackInstances], intervals: List[int]):
        for b in range(len(tracks)):
            scores = torch.max(logits_to_scores(logits=tracks[b].logits), dim=1).values
            is_pos = scores > self.update_threshold
            if self.visualize:
                os.makedirs("./outputs/visualize_tmp/query_updater/", exist_ok=True)
                torch.save(tracks[b].ref_pts.cpu(), "./outputs/visualize_tmp/query_updater/current_ref_pts.tensor")
                # torch.save(tracks[b].query_embed[:, :].cpu(),
                #            "./outputs/visualize_tmp/query_updater/current_query_pos.tensor")
                # torch.save(tracks[b].query_embed[:, :].cpu(),
                #            "./outputs/visualize_tmp/query_updater/current_query_feat.tensor")
                torch.save(tracks[b].query_embed.cpu(),
                           "./outputs/visualize_tmp/query_updater/current_query_feat.tensor")
                torch.save(tracks[b].ids.cpu(), "./outputs/visualize_tmp/query_updater/current_ids.tensor")
                torch.save(tracks[b].labels.cpu(), "./outputs/visualize_tmp/query_updater/current_labels.tensor")
                torch.save(scores.cpu(), "./outputs/visualize_tmp/query_updater/current_scores.tensor")
            if self.use_dab:
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b].boxes[is_pos].detach().clone())
            else:
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b].boxes[is_pos].detach().clone())

            output_pos = pos_to_pos_embed(tracks[b].ref_pts.sigmoid(), num_pos_feats=self.hidden_dim//2)
            output_pos = self.query_pos_head(output_pos)
            output_embed = tracks[b].output_embed

            # Samba
            rate = 1 / intervals[b] if self.fps_invariant else 1  
            # correct formula should be rate = sequence_fps / sampling_interval, but we can assume a constant for a given dataset. 
            # NB: if you are training a general tracker, make sure to use the complete formula aware of the sequence fps  
            hidden_state = tracks[b].hidden_state
            conv_history = tracks[b].conv_history
            output_embed, hidden_state, conv_history = self.samba(
                output_embed.unsqueeze(0),
                hidden_state.unsqueeze(0),
                conv_history.unsqueeze(0),
                output_pos.unsqueeze(0),
                rate=rate)
            tracks[b].hidden_state = hidden_state.squeeze(0)
            tracks[b].conv_history = conv_history.squeeze(0)
            output_embed = output_embed.squeeze(0)
            if self.use_dab:
                if self.with_residual:
                    new_query_embed = self.query_feat_ffn(output_embed)
                    query_embed = tracks[b].query_embed
                    query_embed = query_embed + new_query_embed
                    query_embed = self.norm_emb(query_embed)
                    tracks[b].query_embed[is_pos] = query_embed[is_pos]
                else:
                    tracks[b].query_embed[is_pos] = output_embed[is_pos]
            else:
                if self.with_residual:
                    new_query_embed = self.query_feat_ffn(output_embed)
                    query_embed = tracks[b].query_embed[:, self.hidden_dim:]
                    query_embed = query_embed + new_query_embed
                    query_embed = self.norm_emb(query_embed)
                    tracks[b].query_embed[:, self.hidden_dim:][is_pos] = query_embed[is_pos]
                else:
                    tracks[b].query_embed[:, self.hidden_dim:][is_pos] = output_embed[is_pos]

                # Update query pos, which is not appeared in DAB-D-DETR framework:
                new_query_pos = self.linear_pos2(self.activation(self.linear_pos1(output_embed)))
                query_pos = tracks[b].query_embed[:, :self.hidden_dim]
                query_pos = query_pos + new_query_pos
                query_pos = self.norm_pos(query_pos)
                tracks[b].query_embed[:, :self.hidden_dim][is_pos] = query_pos[is_pos]

            if self.visualize:
                torch.save(tracks[b].ref_pts.cpu(), "./outputs/visualize_tmp/query_updater/next_ref_pts.tensor")
                # torch.save(tracks[b].query_embed[:, :self.hidden_dim].cpu(),
                #            "./outputs/visualize_tmp/query_updater/next_query_pos.tensor")
                # torch.save(tracks[b].query_embed[:, self.hidden_dim:].cpu(),
                #            "./outputs/visualize_tmp/query_updater/next_query_feat.tensor")
                torch.save(tracks[b].query_embed.cpu(),
                           "./outputs/visualize_tmp/query_updater/next_query_feat.tensor")
                torch.save(tracks[b].ids.cpu(), "./outputs/visualize_tmp/query_updater/next_ids.tensor")
                torch.save(tracks[b].labels.cpu(), "./outputs/visualize_tmp/query_updater/next_labels.tensor")
                torch.save(scores.cpu(), "./outputs/visualize_tmp/query_updater/next_scores.tensor")

        return tracks

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
                    fake_tracks.hidden_state = torch.zeros((1, self.hidden_dim * self.expand, self.state_dim), dtype=torch.float, device=device)
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


class MaskedSambaQueryUpdater(SambaQueryUpdater):
    def __init__(self, *args, with_masking: bool = False, with_ref_pts_residual: bool = False, update_only_pos: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_only_pos = update_only_pos
        self.with_masking = with_masking
        self.with_ref_pts_residual = with_ref_pts_residual
        
        if self.with_ref_pts_residual:
            self.bbox_embed = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=4, num_layers=3)
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def update_tracks_embedding(self, tracks: List[TrackInstances], intervals: List[int]):
        for b in range(len(tracks)):
            scores = torch.max(logits_to_scores(logits=tracks[b].logits), dim=1).values
            is_pos = scores > self.update_threshold
            if self.visualize:
                os.makedirs("./outputs/visualize_tmp/query_updater/", exist_ok=True)
                torch.save(tracks[b].ref_pts.cpu(), "./outputs/visualize_tmp/query_updater/current_ref_pts.tensor")
                # torch.save(tracks[b].query_embed[:, :].cpu(),
                #            "./outputs/visualize_tmp/query_updater/current_query_pos.tensor")
                # torch.save(tracks[b].query_embed[:, :].cpu(),
                #            "./outputs/visualize_tmp/query_updater/current_query_feat.tensor")
                torch.save(tracks[b].query_embed.cpu(),
                           "./outputs/visualize_tmp/query_updater/current_query_feat.tensor")
                torch.save(tracks[b].ids.cpu(), "./outputs/visualize_tmp/query_updater/current_ids.tensor")
                torch.save(tracks[b].labels.cpu(), "./outputs/visualize_tmp/query_updater/current_labels.tensor")
                torch.save(scores.cpu(), "./outputs/visualize_tmp/query_updater/current_scores.tensor")
            # rely on previous position for low-confidence boxes
            if self.use_dab:
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b].boxes[is_pos].detach().clone())
            else:
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b].boxes[is_pos].detach().clone())

            output_pos = pos_to_pos_embed(tracks[b].ref_pts.sigmoid(), num_pos_feats=self.hidden_dim//2)
            output_pos = self.query_pos_head(output_pos)
            output_embed = tracks[b].output_embed

            # Mask embeddings and positions of low-confidence boxes (likely occluded)
            if self.with_masking:
                output_embed[~is_pos] = 0.0 * output_embed[~is_pos]
                output_pos[~is_pos] = 0.0 * output_pos[~is_pos]

            # Samba
            rate = 1 / intervals[b] if self.fps_invariant else 1  
            # correct formula should be rate = sequence_fps / sampling_interval, but we can assume a constant for a given dataset. 
            # NB: if you are training a general tracker, make sure to use the complete formula aware of the sequence fps  
            hidden_state = tracks[b].hidden_state
            conv_history = tracks[b].conv_history
            output_embed, hidden_state, conv_history = self.samba(
                output_embed.unsqueeze(0),
                hidden_state.unsqueeze(0),
                conv_history.unsqueeze(0),
                output_pos.unsqueeze(0),
                rate=rate)
            tracks[b].hidden_state = hidden_state.squeeze(0)
            tracks[b].conv_history = conv_history.squeeze(0)
            output_embed = output_embed.squeeze(0)

            # unlike MeMOTR, we update the embed also for low-confidence boxes since Samba takes care of masked observations from context
            update = is_pos if self.update_only_pos else scores >= 0.0
            if self.use_dab:
                if self.with_residual:
                    new_query_embed = self.query_feat_ffn(output_embed)
                    query_embed = tracks[b].query_embed
                    query_embed = query_embed + new_query_embed
                    query_embed = self.norm_emb(query_embed)
                    tracks[b].query_embed[update] = query_embed[update]
                else:
                    tracks[b].query_embed[update] = output_embed[update]
            else:
                if self.with_residual:
                    new_query_embed = self.query_feat_ffn(output_embed)
                    query_embed = tracks[b].query_embed[:, self.hidden_dim:]
                    query_embed = query_embed + new_query_embed
                    query_embed = self.norm_emb(query_embed)
                    tracks[b].query_embed[:, self.hidden_dim:][update] = query_embed[update]
                else:
                    tracks[b].query_embed[:, self.hidden_dim:][update] = output_embed[update]

                # Update query pos, which is not appeared in DAB-D-DETR framework:
                new_query_pos = self.linear_pos2(self.activation(self.linear_pos1(output_embed)))
                query_pos = tracks[b].query_embed[:, :self.hidden_dim]
                query_pos = query_pos + new_query_pos
                query_pos = self.norm_pos(query_pos)
                tracks[b].query_embed[:, :self.hidden_dim][update] = query_pos[update]

            if self.with_ref_pts_residual:
                tmp = self.bbox_embed(output_embed)
                if tracks[b].ref_pts.shape[-1] == 4:
                    tracks[b].ref_pts[is_pos] = tmp[is_pos] + tracks[b].ref_pts[is_pos]
                else:
                    assert tracks[b].ref_pts.shape[-1] == 2
                    tracks[b].ref_pts[is_pos] = tmp[..., :2][is_pos] + tracks[b].ref_pts[is_pos]

            if self.visualize:
                torch.save(tracks[b].ref_pts.cpu(), "./outputs/visualize_tmp/query_updater/next_ref_pts.tensor")
                # torch.save(tracks[b].query_embed[:, :self.hidden_dim].cpu(),
                #            "./outputs/visualize_tmp/query_updater/next_query_pos.tensor")
                # torch.save(tracks[b].query_embed[:, self.hidden_dim:].cpu(),
                #            "./outputs/visualize_tmp/query_updater/next_query_feat.tensor")
                torch.save(tracks[b].query_embed.cpu(),
                           "./outputs/visualize_tmp/query_updater/next_query_feat.tensor")
                torch.save(tracks[b].ids.cpu(), "./outputs/visualize_tmp/query_updater/next_ids.tensor")
                torch.save(tracks[b].labels.cpu(), "./outputs/visualize_tmp/query_updater/next_labels.tensor")
                torch.save(scores.cpu(), "./outputs/visualize_tmp/query_updater/next_scores.tensor")

        return tracks
    

def build(config: dict):
    if config["QUERY_UPDATER"] == "QueryUpdater":
        return QueryUpdater(
                hidden_dim=config["HIDDEN_DIM"],
                ffn_dim=config["FFN_DIM"],
                num_heads=0,
                state_dim=0,
                expand=0,
                num_layers=0,
                conv_dim=0,
                with_self_attn=False,
                dropout=config["DROPOUT"],
                tp_drop_ratio=config["TP_DROP_RATE"] if "TP_DROP_RATE" in config else 0.0,
                fp_insert_ratio=config["FP_INSERT_RATE"] if "FP_INSERT_RATE" in config else 0.0,
                use_checkpoint=config["USE_CHECKPOINT"],
                use_dab=config["USE_DAB"],
                update_threshold=config["UPDATE_THRESH"],
                long_memory_lambda=config["LONG_MEMORY_LAMBDA"],
                visualize=config["VISUALIZE"]
            )
    elif config["QUERY_UPDATER"] == "SambaQueryUpdater":
        return SambaQueryUpdater(
                hidden_dim=config["HIDDEN_DIM"],
                ffn_dim=config["SAMBA_FFN_DIM"],
                num_heads=config["SAMBA_NUM_HEADS"],
                state_dim=config["STATE_DIM"],
                expand=config["EXPAND"],
                num_layers=config["SAMBA_NUM_LAYERS"],
                conv_dim=config["CONV_DIM"],
                with_self_attn=config["WITH_SELF_ATTN"],
                with_self_attn_prior=config["WITH_SELF_ATTN_PRIOR"] if "WITH_SELF_ATTN_PRIOR" in config else False,
                fps_invariant=config["FPS_INVARIANT"],
                dropout=config["DROPOUT"],
                tp_drop_ratio=config["TP_DROP_RATE"] if "TP_DROP_RATE" in config else 0.0,
                fp_insert_ratio=config["FP_INSERT_RATE"] if "FP_INSERT_RATE" in config else 0.0,
                use_checkpoint=config["USE_CHECKPOINT"],
                use_dab=config["USE_DAB"],
                update_threshold=config["UPDATE_THRESH"],
                long_memory_lambda=config["LONG_MEMORY_LAMBDA"],
                with_residual=config["RESIDUAL"],
                visualize=config["VISUALIZE"],
            )
    elif config["QUERY_UPDATER"] == "MaskedSambaQueryUpdater":
        return MaskedSambaQueryUpdater(
                hidden_dim=config["HIDDEN_DIM"],
                ffn_dim=config["SAMBA_FFN_DIM"],
                num_heads=config["SAMBA_NUM_HEADS"],
                state_dim=config["STATE_DIM"],
                expand=config["EXPAND"],
                num_layers=config["SAMBA_NUM_LAYERS"],
                conv_dim=config["CONV_DIM"],
                with_self_attn=config["WITH_SELF_ATTN"],
                with_self_attn_prior=config["WITH_SELF_ATTN_PRIOR"] if "WITH_SELF_ATTN_PRIOR" in config else False,
                fps_invariant=config["FPS_INVARIANT"],
                dropout=config["DROPOUT"],
                tp_drop_ratio=config["TP_DROP_RATE"] if "TP_DROP_RATE" in config else 0.0,
                fp_insert_ratio=config["FP_INSERT_RATE"] if "FP_INSERT_RATE" in config else 0.0,
                use_checkpoint=config["USE_CHECKPOINT"],
                use_dab=config["USE_DAB"],
                update_threshold=config["UPDATE_THRESH"],
                long_memory_lambda=config["LONG_MEMORY_LAMBDA"],
                visualize=config["VISUALIZE"],
                with_masking=config["MASKING"],
                with_residual=config["RESIDUAL"],
                with_ref_pts_residual=config["REF_PTS_RESIDUAL"],
                update_only_pos=config["UPDATE_ONLY_POS"],
            )
    else:
        ValueError(f"Do not support query updater '{config['QUERY_UPDATER']}'")

