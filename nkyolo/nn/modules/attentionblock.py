# nkyolo/nn/modules/attentionblock.py

import jittor as jt
import jittor.nn as nn
import copy

# ================================
# 多头注意力机制 (MultiheadAttention)
# ================================

class MultiheadAttention(nn.Module):
    """
    Jittor 实现的多头注意力机制，功能对齐 PyTorch 的 nn.MultiheadAttention。

    参数:
        embed_dim: 输入特征维度
        num_heads: 注意力头的数量
        dropout: dropout概率，默认为0.0
        batch_first: 如果为True，输入输出为 (batch, seq_len, embed_dim)
        bias: 是否在线性层中使用偏置，默认为 True
        average_attn_weights: 是否对多头注意力权重取平均，默认为 True
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=False, bias=True, average_attn_weights=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.average_attn_weights = average_attn_weights
        self.bias = bias

        assert embed_dim % num_heads == 0, "嵌入维度必须能被头数整除"
        self.head_dim = embed_dim // num_heads

        # Q, K, V 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        """初始化参数"""
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
        assert embed_dim == self.embed_dim, f"输入维度 {embed_dim} 不匹配 {self.embed_dim}"

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 分头: (L, B, E) -> (B, H, L, D)
        q = q.view(tgt_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(src_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 3, 0)  # (B, H, D, S)
        v = v.view(src_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # (B, H, S, D)

        scaling = float(self.head_dim) ** -0.5
        attn_scores = jt.matmul(q, k) * scaling  # (B, H, L, S)

        # 应用 attn_mask
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1,1,L,S)
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)  # (B,1,L,S)
            else:
                raise ValueError(f"attn_mask 维度应为 2 或 3，但得到 {attn_mask.ndim}")
            min_value = -1e8
            attn_scores = jt.where(attn_mask.bool(), attn_scores, min_value)

        # 应用 key_padding_mask
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
            mask = mask.broadcast_shape((batch_size, 1, tgt_len, src_len))
            min_value = -1e8
            attn_scores = jt.where(mask, min_value, attn_scores)

        attn_weights = nn.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # 加权求和
        attn_output = jt.matmul(attn_weights, v)  # (B, H, L, D)

        # 合并头
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
    # 移除 __getstate__ / __setstate__ 避免 deepcopy 错误
    # ========================
    # ❌ 不要实现 __getstate__ 调用子模块 __getstate__
    # 已删除，使用默认行为即可

