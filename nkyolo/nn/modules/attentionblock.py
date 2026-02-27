# nkyolo/nn/modules/attentionblock.py

import jittor as jt
import jittor.nn as nn

# ================================
# Multihead attention mechanism (MultiheadAttention)
# ================================

class MultiheadAttention(nn.Module):
    """
    Jittor implementation of multihead attention, aligned with PyTorch nn.MultiheadAttention.

    Args:
        embed_dim: input feature dimension
        num_heads: number of attention heads
        dropout: dropout probability, default 0.0
        batch_first: if True, input/output are (batch, seq_len, embed_dim)
        bias: whether to use bias in linear layers, default True
        average_attn_weights: whether to average attention weights across heads, default True
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=False, bias=True, average_attn_weights=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.average_attn_weights = average_attn_weights
        self.bias = bias

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embed_dim // num_heads

        # Q, K, V projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.bias:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def execute(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        if self.batch_first:
            query = query.permute(1, 0, 2)
            key = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        tgt_len, batch_size, embed_dim = query.shape
        src_len = key.shape[0]
        assert embed_dim == self.embed_dim, f"Input dimension {embed_dim} mismatch {self.embed_dim}"

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Split heads: (L, B, E) -> (B, H, L, D)
        q = q.view(tgt_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(src_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 3, 0)  # (B, H, D, S)
        v = v.view(src_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # (B, H, S, D)

        scaling = float(self.head_dim) ** -0.5
        attn_scores = jt.matmul(q, k) * scaling  # (B, H, L, S)

        # Apply attn_mask
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1,1,L,S)
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)  # (B,1,L,S)
            else:
                raise ValueError(f"attn_mask must have 2 or 3 dims, got {attn_mask.ndim}")
            min_value = -1e8
            attn_scores = jt.where(attn_mask.bool(), attn_scores, min_value)

        # Apply key_padding_mask
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
            mask = mask.broadcast_shape((batch_size, 1, tgt_len, src_len))
            min_value = -1e8
            attn_scores = jt.where(mask, min_value, attn_scores)

        attn_weights = nn.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Weighted sum
        attn_output = jt.matmul(attn_weights, v)  # (B, H, L, D)

        # Merge heads
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(tgt_len, batch_size, -1)
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.permute(1, 0, 2)  # (B, L, E)

        if need_weights:
            if self.average_attn_weights and self.num_heads > 1:
                weights = attn_weights.mean(dim=1)  # (B, L, S)
            else:
                weights = attn_weights  # (B, H, L, S)
            return attn_output, weights
        else:
            return attn_output

    # ========================
    # Remove __getstate__ / __setstate__ to avoid deepcopy errors
    # ========================
    # Do not implement __getstate__ calling submodules' __getstate__
    # Removed; default behavior is sufficient
