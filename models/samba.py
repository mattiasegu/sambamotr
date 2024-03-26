"""Simple, minimal implementation of Samba in one file of PyTorch.
Samba is a Sequence-interactive Mamba model that allows for cross-sequence communication 
while predicting the next token for each sequence.

This implementation builds on top of the minimal Mamba implementation from 
    https://github.com/johnma2006/mamba-minimal/tree/master

References:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")
"""
from __future__ import annotations
import math
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from mmdet.registry import MODELS
from mmdet.utils import OptConfigType


class SambaBlock(nn.Module):
    """A single Samba block.
    
    This module implements Samba (Synchronized Mamba), which allows for
    syncronization across the hidden states of multiple instances of Mamba
    applied to different sequences. Samba effectively binds the prediction
    of Mamba for a given sequence to that of the other sequences being
    processed.

    Args:
        d_model (int): input dim. Defaults to 256.
        d_state (int): latent state dim. Defaults to 16. 
        expand (int): expansion factor. Defaults to 2.
        dt_rank (Union[int, str]): if int, it sets the rank of Δ. If str,
            it defines the policy for computing the rank of Δ. 
            Defaults to 'auto'.
        d_conv (int): conv layer dim. Defaults to 4.
        conv_bias (bool): whether conv layer has bias term. Defaults to True.
        bias (bool): whether linear layers have bias term. Defaults to True.
        with_self_attn (bool): whether to use self attention layers to
            syncronize multiple Mamba instances. Defaults to True.
        self_attn_cfg (:obj:`ConfigDict` or dict, optional):
            Config for self-attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `RMSN`, following the original Mamba.

    Notation:
        n or d_state: latent state dim      (`N` in [1] Algorithm 2)
        b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
        k: numer of sequences
        d or d_model: hidden dim
        n or d_state: latent state dim      (`N` in [1] Algorithm 2)
        expand: expansion factor            (`E` in [1] Section 3.4)
        d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
        A, B, C, D: state space parameters  (See any state space representation formula)
        Δ or delta: input-dependent step size
        dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

    References:
        [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
            https://arxiv.org/abs/2312.00752
    """
    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 expand: int = 2,
                 dt_rank: Union[int, str] = 'auto',
                 d_conv: int = 4,
                 conv_bias: bool = True,
                 bias: bool = False,
                 with_self_attn: bool = True,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256, num_heads=8, dropout=0.0),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,  # TODO: expand by d_state factor?
                     feedforward_channels=1024,  # TODO: expand by d_state factor?
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: OptConfigType = dict(type='RMSN'),
        ):
        super().__init__()
        
        d_inner = int(expand * d_model)

        self.d_model = d_model
        self.expand = expand
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        
        if dt_rank == 'auto':
            dt_rank = math.ceil(self.d_model / 16)
        self.dt_rank = dt_rank

        self.with_self_attn = with_self_attn

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_inner,
            # padding=d_conv - 1,
            padding=0,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

        # self_attn allows for cross-sequences syncronization of Mamba
        expansion_factor = expand * d_state
        if with_self_attn and self_attn_cfg is not None:
            self_attn_cfg = self_attn_cfg.copy()
            self_attn_cfg.embed_dims *= expansion_factor
            self.self_attn = MultiheadAttention(**self_attn_cfg)

        if ffn_cfg is not None:
            ffn_cfg = ffn_cfg.copy()
            ffn_cfg.embed_dims *= expansion_factor
            ffn_cfg.feedforward_channels *= expansion_factor
            self.ffn = FFN(**ffn_cfg)

        norms_list = [
            build_norm_layer(norm_cfg, d_model * expansion_factor)[1] 
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self, inputs, hidden_states, conv_history):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Notation:
            b: batch size                       
            k: number of sequences
            d or d_model: hidden dim

        Args:
            inputs: shape (b, k, d)
            hidden_states: shape (b, d, n)
            conv_history: shape (b, k, l, d), where l is the number of 
                historical conv states conserved in the history. By default,
                Samba assumes that l == d_conv, since we only need to preserve a
                d_conv-sized history to perform the convolution.
    
        Returns:
            output: shape (b, k, d)
            conv_history: shape (b, k, l, d), the updated conv_history. 
                Compared to the input conv_history, it has been shifted to the
                left by 1 place along the l dimension and the last element along
                the l dimension has been replaced by the current conv state. 
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
        """
        (b, k, l, d) = conv_history.shape
        assert l == self.d_conv, 'By default, Samba assumes that ' \
            'l == d_conv, since we only need to preserve a d_conv-sized ' \
            'history to perform the convolution.'
        
        inputs_and_res = self.in_proj(inputs)  # shape (b, k, 2 * d_in)
        (inputs, res) = inputs_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        # Update conv history
        conv_state = inputs
        conv_history = torch.roll(conv_history, -1, -2)  # roll left on l dim
        conv_history[:,:,-1,:] = conv_state

        # Process conv history
        inputs = rearrange(conv_history, 'b k l d_in -> (b k) d_in l')
        inputs = self.conv1d(inputs)[..., 0]  # causal conv 
        inputs = rearrange(inputs, '(b k) d_in -> b k d_in', b=b, k=k)
        
        inputs = F.silu(inputs)

        outputs, hidden_states = self.ssm(inputs, hidden_states)
        
        outputs = outputs * F.silu(res)
        
        outputs = self.out_proj(outputs)

        return outputs, hidden_states, conv_history
    
    def ssm(self, inputs, hidden_states):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, k, d_in)    (See Glossary at top for definitions of b, k, d_in, n...)
    
        Returns:
            output: shape (b, k, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        inputs_dbl = self.x_proj(inputs)  # (b, k, dt_rank + 2*n)
        
        (delta, B, C) = inputs_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, k, dt_rank). B, C: (b, k, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        outputs, hidden_states = self.single_selective_scan(inputs, hidden_states, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return outputs, hidden_states
    
    def single_selective_scan(self, u, x, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
        """
        (b, k, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b k d_in, d_in n -> b k d_in n'))
        deltaB_u = einsum(delta, B, u, 'b k d_in, b k n, b k d_in -> b k d_in n')
        
        # Perform one pass of the selective scan (see scan_SSM() in The Annotated S4 [2])
        x = deltaA * x + deltaB_u

        # Syncronize Mambas
        x = rearrange(x, 'b k d_in n -> b k (d_in n)')
        if self.with_self_attn:
            x = self.self_attn(query=x, key=x, value=x)
            x = self.norms[0](x)
            x = self.ffn(x)
            x = self.norms[1](x)
        x = rearrange(x, 'b k (d_in n) -> b k d_in n', d_in=d_in, n=n)

        # Predict
        y = einsum(x, C, 'b k d_in n, b k n -> b k d_in')

        y = y + u * D

        return y, x


class ResidualBlock(nn.Module):
    """Simple block wrapping a Samba block with normalization and residual connection.
    
    Args:
        d_model (int): input dim. Defaults to 256.
        d_state (int): latent state dim. Defaults to 16. 
        expand (int): expansion factor. Defaults to 2.
        dt_rank (Union[int, str]): if int, it sets the rank of Δ. If str,
            it defines the policy for computing the rank of Δ. 
            Defaults to 'auto'.
        d_conv (int): conv layer dim. Defaults to 4.
        conv_bias (bool): whether conv layer has bias term. Defaults to True.
        bias (bool): whether linear layers have bias term. Defaults to True.
        self_attn_cfg (:obj:`ConfigDict` or dict, optional):
            Config for self-attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `RMSN`, following the original Mamba.
    """
    def __init__(self, 
                 d_model: int = 256,
                 d_state: int = 16,
                 expand: int = 2,
                 dt_rank: Union[int, str] = 'auto',
                 d_conv: int = 4,
                 conv_bias: bool = True,
                 bias: bool = False,
                 with_self_attn=True,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256, num_heads=8, dropout=0.0),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: OptConfigType = dict(type='RMSN'),
    ) -> None:
        super().__init__()

        self.mixer = SambaBlock(d_model,
                                d_state,
                                expand,
                                dt_rank,
                                d_conv,
                                conv_bias,
                                bias,
                                with_self_attn,
                                self_attn_cfg,
                                ffn_cfg)
        self.norm = build_norm_layer(norm_cfg, d_model)[1]
        
    def forward(self, inputs, hidden_states, conv_history):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        outputs, hidden_states, conv_history = self.mixer(inputs, hidden_states, conv_history)  # unlike Mamba, the input is already normalized here
        outputs = outputs + inputs

        return outputs, hidden_states, conv_history


@MODELS.register_module()
class Samba(nn.Module):
    """Full Samba model.

    This module implements the full Samba (Synchronized Mamba) model, which
    consists of a stack of residual Samba block. When processing multiple
    sequences simultaneously, Samba syncronizes the hidden states of the
    multiple Mamba instances that are independently processing each input 
    sequence. Samba effectively binds the prediction of Mamba for each 
    input sequence to that of the other sequences being processed.

    Args:
        num_layers (int): number of Samba layers. Defaults to 2.
        d_model (int): model dim. Defaults to 64.
        d_state (int): latent state dim. Defaults to 16. 
        expand (int): expansion factor. Defaults to 2.
        dt_rank (Union[int, str]): if int, it sets the rank of Δ. If str,
            it defines the policy for computing the rank of Δ. 
            Defaults to 'auto'.
        d_conv (int): conv layer dim. Defaults to 4.
        conv_bias (bool): whether conv layer has bias term. Defaults to True.
        bias (bool): whether linear layers have bias term. Defaults to True.
        self_attn_cfg (:obj:`ConfigDict` or dict, optional):
            Config for self-attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `RMSN`, following the original Mamba.
    """
    def __init__(self, 
                 num_layers: int = 2,
                 norm_cfg: OptConfigType = dict(type='RMSN'),
                 d_model=64,
                 layer_cfg: OptConfigType = dict(
                     d_model=64,
                     d_state=16,
                     expand=2,
                     dt_rank='auto',
                     d_conv=4,
                     conv_bias=True,
                     bias=False,
                     with_self_attn=True,
                     self_attn_cfg=dict(
                         embed_dims=64, num_heads=8, dropout=0.0),
                     ffn_cfg=dict(
                         embed_dims=64,
                         feedforward_channels=256,
                         num_fcs=2,
                         ffn_drop=0.,
                         act_cfg=dict(type='ReLU', inplace=True)),
                     norm_cfg=dict(type='RMSN')),
        ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList([ResidualBlock(**layer_cfg)
                                     for _ in range(num_layers)])
        self.norm_f = build_norm_layer(norm_cfg, d_model)[1]

    def forward(self, query, hidden_states, conv_history, query_pos=None):
        """
        Args:
            query (long tensor): shape (b, k, d_in), where d_in == d_model
            hidden_states: shape (b, k, d, d_state)
            conv_history: shape (b, k, num_layers, l, d), the per-layer history of
                the last l conv states. By default, Samba assumes that
                l == d_conv, since we only need to preserve a d_conv-sized
                history to perform the convolution.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`. If not None, it will
                be added to `query` before forward function. Defaults to None.
    
        Returns:
            output: shape (b, k, d_in)
            hidden_states: shape (b, k, d, d_state)
            conv_history: shape (b, k, num_layers, l, d), the updated per-layer
                history of the last l conv states.
        """
        
        out_conv_history = conv_history.clone()  # TODO: needed?
        for i, layer in enumerate(self.layers):
            if query_pos is not None:
                query = query + query_pos
            _conv_history = conv_history[:,:,i,:,:]
            query, hidden_states, _conv_history = layer(
                query, hidden_states, _conv_history)
            out_conv_history[:,:,i,:,:] = _conv_history
        
        outputs = self.norm_f(query)

        return outputs, hidden_states, out_conv_history

