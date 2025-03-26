"""Simple, minimal implementation of Samba in one file of PyTorch.
Samba is a set-of-sequences model based on Synchronized Mamba models. 
Synchronization allows for cross-sequence communication while predicting
the next token for each sequence.

Samba was introduced in the paper "Samba: Synchronized Set-of-Sequences
Modeling for Multiple Object Tracking" by Mattia Segu et al. (2024).
The paper is available at https://arxiv.org/abs/2410.01806

This implementation builds on top of the minimal Mamba implementation from 
    https://github.com/johnma2006/mamba-minimal/tree/master

References:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4
    [3] Samba: Synchronized Set-of-Sequences Modeling for Multiple Object Tracking (Mattia Segu et al.)
        https://arxiv.org/abs/2410.01806

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    k: numer of sequences
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

from einops import rearrange, repeat, einsum
from typing import Union

from ..ffn import FFN
from .attention import MultiheadAttention


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
        with_self_attn_prior (bool): whether to apply self attention only on 
            the prior Ax. Defaults to False.
        self_attn_cfg (:obj:`ConfigDict` or dict, optional):
            Config for self-attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        with_conv (bool): whether to use a causal conv1d layer. Defaults to True.
        inner_layernorms (bool): whether to apply layer norms to the dt, B, C 
            as in Jamba. Defaults to False.
        with_input_layernorm (bool): whether to apply input layernorm.
            Defaults to True.
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
                 with_self_attn_prior: bool = False,
                 num_attn_layers: int = 1,  # Number of self-attention layers
                 self_attn_cfg: dict = dict(
                     embed_dims=256, num_heads=8, dropout=0.0),
                 ffn_cfg: dict = dict(
                     embed_dims=256,  # gets expanded by d_state factor
                     feedforward_channels=1024,  # gets expanded by d_state factor
                     ffn_drop=0.),
                 with_conv: bool = True,
                 inner_layernorms: bool = False,
                 with_input_layernorm: bool = True,
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
        self.with_self_attn_prior = with_self_attn_prior

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)

        # optionally skip causal conv1d (unused)
        self.with_conv = with_conv
        if self.with_conv:
            self.conv1d = nn.Conv1d(
                in_channels=d_inner,
                out_channels=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=d_inner,
                padding=0,
            )
        else:
            self.conv1d = None

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in. 
        # the bias represents the learnable component of delta that is NOT input-dependent
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

        # Stack of self-attention layers implementing synchronization
        if with_self_attn and self_attn_cfg is not None:
            self_attn_cfg = self_attn_cfg.copy()
            self_attn_cfg['embed_dims'] = d_state  # Use d_state for self-attn

            # Create multiple attention layers
            self.self_attn_layers = nn.ModuleList([
                MultiheadAttention(
                    embed_dim=self_attn_cfg['embed_dims'],
                    num_heads=self_attn_cfg['num_heads'],
                    batch_first=True,
                )
                for _ in range(num_attn_layers)
            ])

            # Add FFNs and layer norms for each attention layer
            self.ffn_layers = nn.ModuleList([
                FFN(d_model=self_attn_cfg['embed_dims'],
                    d_ffn=ffn_cfg['feedforward_channels'],
                    dropout=ffn_cfg['ffn_drop'],
                    bias=True)
                for _ in range(num_attn_layers)
            ])
            self.pre_norms = nn.ModuleList([
                nn.LayerNorm(self_attn_cfg['embed_dims'])
                for _ in range(num_attn_layers)  # Norms for attention and FFN
            ])
            self.norms = nn.ModuleList([
                nn.LayerNorm(self_attn_cfg['embed_dims'])
                for _ in range(num_attn_layers)  # Norms for attention and FFN
            ])

        # Optional samba input layernorm
        self.with_input_layernorm = with_input_layernorm
        if self.with_input_layernorm:
            self.input_layernorm = nn.LayerNorm(d_inner)  # Initialize LayerNorm
        else:
            self.input_layernorm = None

        # Used in jamba
        self.inner_layernorms = inner_layernorms
        if self.inner_layernorms:
            self.dt_layernorm = nn.LayerNorm(dt_rank)
            self.B_layernorm = nn.LayerNorm(d_state)
            self.C_layernorm = nn.LayerNorm(d_state)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        # Learnable position embedding scaling, initialized from exp(0) = 1
        self.pos_emb_scaling = nn.Parameter(torch.zeros(1))  # Exponential scaling factor
        self.sync_scaling = nn.Parameter(torch.zeros(1))  # Exponential scaling factor

        # Learnable position embedding for channel-wise efficient synchronization
        self.position_embedding = nn.Parameter(torch.randn(d_inner, d_state))

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, inputs, hidden_states, conv_history, rate=1):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Notation:
            b: batch size                       
            k: number of sequences
            d or d_model: hidden dim

        Args:
            inputs: shape (b, k, d)
            hidden_states: shape (b, k, d, n)
            conv_history: shape (b, k, l, d), where l is the number of 
                historical conv states conserved in the history. By default,
                Samba assumes that l == d_conv, since we only need to preserve a
                d_conv-sized history to perform the convolution.
            rate (float): the rate at which to apply the SSM. In case of video
                processing, it corresponds to the frame rate. It can be useful
                in case different frame rates are observed between training and
                inference time. However, we do not use it in the final paper and
                set rate=1 always. Defaults to 1.
    
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
        if self.with_conv:
            inputs = self.conv1d(inputs)[..., 0]  # causal conv 
        else:
            inputs = inputs[:,:,-1]  # skip causal conv 
        inputs = rearrange(inputs, '(b k) d_in -> b k d_in', b=b, k=k)
        
        inputs = F.silu(inputs)
        if self.input_layernorm is not None:
            inputs = self.input_layernorm(inputs)

        outputs, hidden_states = self.ssm(inputs, hidden_states, rate)
        
        outputs = outputs * F.silu(res)
        
        outputs = self.out_proj(outputs)

        return outputs, hidden_states, conv_history
    
    def ssm(self, inputs, hidden_states, rate=1):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            inputs: shape (b, k, d)
            hidden_states: shape (b, k, d, n)
            rate (float): the rate at which to apply the SSM. In case of video
                processing, it corresponds to the frame rate. It can be useful
                in case different frame rates are observed between training and
                inference time. However, we do not use it in the final paper and
                set rate=1 always. Defaults to 1.
    
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
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta)) / rate  # (b, k, d_in)

        outputs, hidden_states = self.single_selective_scan(inputs, hidden_states, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return outputs, hidden_states

    def sync_hidden_states(self, x):
        """Synchronize hidden states with multiple self-attention layers. 
        This is a key innovation in Samba. See Section 4.2 in the Samba paper [3].
        
        Notice that to efficiently synchronize hidden states across multiple sequences,
        we apply synchronization channel-wise (each dimension d_in is synchronized independently).
        To leverage a single set of synchronization weights, we learn a position embedding for each channel d_in.

        To efficiently synchronize hidden states across k multiple sequences,
        synchronization is applied **channel-wise** (each dimension `d_in` is synchronized independently).
        To achieve this efficiently and in parallel, a single set of synchronization weights is leveraged
        by learning a position embedding for each channel `d_in`. Each token in the sequence is then of 
        dimension n, which defaults to 16.
        
        Args:
            x: shape (b, k, d, n)
    
        Returns:
            output: shape (b, k, d, n)
        """
        b, k, d_in, n = x.shape
        x = rearrange(x, 'b k d_in n -> (b d_in) k n')
        pos_emb = repeat(self.position_embedding, 'd_in n -> (b d_in) k n', b=b, k=k)
        pos_emb = pos_emb * self.pos_emb_scaling.exp()

        for i, attn_layer in enumerate(self.self_attn_layers):
            res_attn = x
            x = self.pre_norms[i](x)  # Pre-norm
            # Set need_weights to False to avoid computing attention weights for faster scaled_dot_product_attention
            x, weight = attn_layer(query=x+pos_emb, key=x+pos_emb, value=res_attn, need_weights=False)  # Apply attention
            x = self.norms[i](x + res_attn)  # Add residual and normalize
            x = self.ffn_layers[i](x)  # FFN already includes residual and norm

        x = rearrange(x, '(b d_in) k n -> b k d_in n', b=b, d_in=d_in)
        return x

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
            u: shape (b, k, d_in)    (See Glossary at top for definitions of b, k, l, d_in, n...)
            x: shape (b, k, d_in, n) 
            delta: shape (b, k, d_in)
            A: shape (d_in, n)
            B: shape (b, k, n)
            C: shape (b, k, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, k, d_in)
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
        if self.with_self_attn:
            if self.with_self_attn_prior:
                x = deltaA * x
                x = x + self.sync_scaling.exp() * self.sync_hidden_states(x)
                x = x + deltaB_u
            else:
                x = deltaA * x + deltaB_u
                x = x + self.sync_scaling.exp() * self.sync_hidden_states(x)
        else:
            x = deltaA * x + deltaB_u

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
        with_conv (bool): whether to use a causal conv1d layer. Defaults to True.
        inner_layernorms (bool): whether to apply layer norms to the dt, B, C 
            as in Jamba. Defaults to False.
        with_input_layernorm (bool): whether to apply input layernorm.
            Defaults to True.
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
                 with_self_attn_prior=False,
                 num_attn_layers: int = 1,  # Number of self-attention layers
                 self_attn_cfg: dict = dict(
                     embed_dims=256, num_heads=8, dropout=0.0),
                 ffn_cfg: dict = dict(
                     embed_dims=256,  # gets expanded by d_state factor
                     feedforward_channels=1024,  # gets expanded by d_state factor
                     ffn_drop=0.),
                 with_conv: bool = True,
                 inner_layernorms: bool = True,
                 with_input_layernorm: bool = True,
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
                                with_self_attn_prior,
                                num_attn_layers,
                                self_attn_cfg,
                                ffn_cfg,
                                with_conv,
                                inner_layernorms,
                                with_input_layernorm)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, inputs, hidden_states, conv_history, rate):
        """
        Args:
            query (long tensor): shape (b, k, d_in), where d_in == d_model
            hidden_states: shape (b, k, d, n)
            conv_history: shape (b, k, l, d), the per-layer history of
                the last l conv states. By default, Samba assumes that
                l == d_conv, since we only need to preserve a d_conv-sized
                history to perform the convolution.
            rate (float): the rate at which to apply the SSM. In case of video
                processing, it corresponds to the frame rate. It can be useful
                in case different frame rates are observed between training and
                inference time. However, we do not use it in the final paper and
                set rate=1 always. Defaults to 1.
    
        Returns:
            output: shape (b, k, d_in)
            hidden_states: shape (b, k, num_layers, d, n)
            conv_history: shape (b, k, num_layers, l, d), the updated per-layer
                history of the last l conv states.

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        outputs, hidden_states, conv_history = self.mixer(self.norm(inputs), hidden_states, conv_history, rate)
        outputs = outputs + inputs

        return outputs, hidden_states, conv_history


class Samba(nn.Module):
    """Full Samba model.

    This module implements the full Samba (Synchronized Mamba) model, which
    consists of a stack of residual Samba block. When processing multiple
    sequences simultaneously, Samba syncronizes the hidden states of the
    multiple Mamba instances that are independently processing each input 
    sequence. Samba effectively binds the prediction of Mamba for each 
    input sequence to that of the other sequences being processed.

    Args:
        num_layers (int): number of Samba layers. Defaults to 4.
        d_model (int): model dim. Defaults to 256.
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
        with_conv (bool): whether to use a causal conv1d layer. Defaults to True.
        inner_layernorms (bool): whether to apply layer norms to the dt, B, C 
            as in Jamba. Defaults to False.
        with_input_layernorm (bool): whether to apply input layernorm.
            Defaults to True.
    """
    def __init__(self, 
                 num_layers: int = 4,
                 d_model=256,
                 layer_cfg: dict = dict(
                     d_model=256,
                     d_state=16,
                     expand=2,
                     dt_rank='auto',
                     d_conv=4,
                     conv_bias=True,
                     bias=False,
                     with_self_attn=True,
                     with_self_attn_prior=False,
                     num_attn_layers=1,  # Number of self-attention layers
                     self_attn_cfg=dict(
                         embed_dims=256, num_heads=8, dropout=0.0),
                     ffn_cfg=dict(
                         embed_dims=256,  # gets expanded by d_state factor
                         feedforward_channels=1024,  # gets expanded by d_state factor
                         ffn_drop=0.),
                     with_conv=True,
                     inner_layernorms=True,
                     with_input_layernorm=True,),
        ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList([ResidualBlock(**layer_cfg)
                                     for _ in range(num_layers)])
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, query, hidden_states, conv_history, query_pos=None, rate=1):
        """
        Args:
            query (long tensor): shape (b, k, d_in), where d_in == d_model
            hidden_states: shape (b, k, num_layers, d, d_state)
            conv_history: shape (b, k, num_layers, l, d), the per-layer history of
                the last l conv states. By default, Samba assumes that
                l == d_conv, since we only need to preserve a d_conv-sized
                history to perform the convolution.
            query_pos (Tensor): The positional encoding for query, with
                the same shape (b, k, d_in) as `query`. If not None, it will
                be added to `query` before forward function. Defaults to None.
            rate (float): the rate at which to apply the SSM. In case of video
                processing, it corresponds to the frame rate. It can be useful
                in case different frame rates are observed between training and
                inference time. However, we do not use it in the final paper and
                set rate=1 always. Defaults to 1.
    
        Returns:
            output: shape (b, k, d_in)
            hidden_states: shape (b, k, num_layers, d, n)
            conv_history: shape (b, k, num_layers, l, d), the updated per-layer
                history of the last l conv states.
        """
        out_conv_history = conv_history.clone()
        out_hidden_states = hidden_states.clone()
        for i, layer in enumerate(self.layers):
            if query_pos is not None:
                query = query + query_pos
            _conv_history = conv_history[:,:,i,:,:]
            _hidden_states = hidden_states[:,:,i,:,:]
            query, _hidden_states, _conv_history = layer(
                query, _hidden_states, _conv_history, rate)
            out_conv_history[:,:,i,:,:] = _conv_history
            out_hidden_states[:,:,i,:,:] = _hidden_states
        
        outputs = self.norm_f(query)

        return outputs, out_hidden_states, out_conv_history

