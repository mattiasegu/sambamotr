
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, Tuple

def scaled_dot_product_attention(query, key, value, attn_mask=None, key_padding_mask=None, dropout_p=0.0):
    """
    Computes the scaled dot-product attention.
    """
    d_k = query.shape[-1]  # Key dimension
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=query.dtype))

    # Apply attention mask (if any)
    if attn_mask is not None:
        attn_scores += attn_mask

    # Apply key padding mask (if any)
    if key_padding_mask is not None:
        attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

    # Compute attention weights
    attn_weights = F.softmax(attn_scores, dim=-1)

    # Apply dropout
    if dropout_p > 0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    # Compute output
    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Input projection matrices (Q, K, V)
        self.q_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.k_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.v_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))

        self.q_proj_bias = nn.Parameter(torch.empty(embed_dim))
        self.k_proj_bias = nn.Parameter(torch.empty(embed_dim))
        self.v_proj_bias = nn.Parameter(torch.empty(embed_dim))

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_p = dropout
        self.batch_first = batch_first

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        
        nn.init.zeros_(self.q_proj_bias)
        nn.init.zeros_(self.k_proj_bias)
        nn.init.zeros_(self.v_proj_bias)

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Custom implementation of Multihead Attention.
        """

        # Ensure batch-first format
        if not self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        batch_size, tgt_len, embed_dim = query.shape
        src_len = key.shape[1]

        # Compute Q, K, V with separate projections
        q = F.linear(query, self.q_proj_weight, self.q_proj_bias)
        k = F.linear(key, self.k_proj_weight, self.k_proj_bias)
        v = F.linear(value, self.v_proj_weight, self.v_proj_bias)

        # Correct reshaping for multi-head attention: [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, self.num_heads, tgt_len, self.head_dim)
        k = k.view(batch_size, self.num_heads, src_len, self.head_dim)
        v = v.view(batch_size, self.num_heads, src_len, self.head_dim)

        # Compute attention
        if need_weights:
            attn_output, attn_weights = scaled_dot_product_attention(q, k, v, attn_mask, key_padding_mask, self.dropout_p)
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,             # shape must be broadcastable to (T, S) or (B*nH, T, S)
                dropout_p=self.dropout_p,
            )
            attn_weights = None

        # Reshape back: [batch, tgt_len, embed_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, tgt_len, embed_dim)

        # Apply final linear projection
        attn_output = self.out_proj(attn_output)

        # No need to swap back the dimensions at the end
        if not self.batch_first:
            attn_output = attn_output.transpose(1, 0)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)  # Average over heads
            return attn_output, attn_weights
        else:
            return attn_output, None
