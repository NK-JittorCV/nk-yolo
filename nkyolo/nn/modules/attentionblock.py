import jittor as jt
import jittor.nn as nn


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
        """初始化参数，参考 PyTorch 实现"""
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
        """
        执行多头注意力

        参数:
            query: (L, B, E) 或 (B, L, E)
            key: (S, B, E) 或 (B, S, E)
            value: (S, B, E) 或 (B, S, E)
            key_padding_mask: (B, S) bool 或 0/1 张量，表示哪些位置是填充
            need_weights: 是否返回注意力权重
            attn_mask: (L, S) 或 (B, L, S) 的掩码张量

        返回:
            attn_output: 输出张量，形状与 query 相同
            attn_output_weights: 注意力权重 (B, L, S) 或 (B, H, L, S)
        """
        # 处理 batch_first
        if self.batch_first:
            query = query.permute(1, 0, 2)  # -> (L, B, E)
            key = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        tgt_len, batch_size, embed_dim = query.shape
        src_len = key.shape[0]
        assert embed_dim == self.embed_dim, f"输入维度 {embed_dim} 不匹配 {self.embed_dim}"

        # 线性变换
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 分头: (L, B, E) -> (B, H, L, D)
        q = q.view(tgt_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(src_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 3, 0)  # (B, H, D, S)
        v = v.view(src_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # (B, H, S, D)

        # 计算注意力分数: (B, H, L, S)
        scaling = float(self.head_dim) ** -0.5
        attn_scores = jt.matmul(q, k) * scaling

        # 应用 attn_mask
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                # (L, S) -> (1, 1, L, S)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.ndim == 3:
                # (B, L, S) -> (B, 1, L, S)
                attn_mask = attn_mask.unsqueeze(1)
            else:
                raise ValueError(f"attn_mask 维度应为 2 或 3，但得到 {attn_mask.ndim}")
            # 使用极小值替代 -inf（Jittor 不稳定支持 inf）
            min_value = -1e8
            attn_scores = jt.where(attn_mask.bool(), attn_scores, min_value)

        # 应用 key_padding_mask: (B, S) -> (B, 1, 1, S)
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            mask = mask.broadcast_shape((batch_size, 1, tgt_len, src_len))  # -> (B, 1, L, S)
            min_value = -1e8
            attn_scores = jt.where(mask, min_value, attn_scores)

        # Softmax + Dropout
        attn_weights = nn.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # 加权求和: (B, H, L, S) @ (B, H, S, D) -> (B, H, L, D)
        attn_output = jt.matmul(attn_weights, v)

        # 合并头: (B, H, L, D) -> (L, B, E)
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(tgt_len, batch_size, -1)
        attn_output = self.out_proj(attn_output)

        # 恢复 batch_first
        if self.batch_first:
            attn_output = attn_output.permute(1, 0, 2)  # (B, L, E)

        # 返回权重
        if need_weights:
            if self.average_attn_weights and self.num_heads > 1:
                weights = attn_weights.mean(dim=1)  # (B, L, S)
            else:
                weights = attn_weights  # (B, H, L, S)
            return attn_output, weights
        else:
            return attn_output

    # def __getstate__(self):
    #     """用于序列化模型状态（如保存）"""
    #     state = self.__dict__.copy()
    #     # 确保子模块正确序列化
    #     state['q_proj'] = self.q_proj.__getstate__()
    #     state['k_proj'] = self.k_proj.__getstate__()
    #     state['v_proj'] = self.v_proj.__getstate__()
    #     state['out_proj'] = self.out_proj.__getstate__()
    #     state['dropout_layer'] = self.dropout_layer.__getstate__()
    #     return state

    def __getstate__(self):
    # 只保存 __dict__，不要调用子模块的 __getstate__
        return self.__dict__.copy()

    def __setstate__(self, state):
    # 恢复 __dict__
        self.__dict__.update(state)


def multi_scale_deformable_attn_pytorch(
    value: jt.Var,
    value_shapes: list[tuple[int, int]],
    sampling_locations: jt.Var,
    attention_weights: jt.Var
) -> jt.Var:
    """
    多尺度可变形注意力机制的Jittor实现。

    Args:
        value: (batch_size, num_heads, num_value, value_dim)
        value_shapes: [(h1, w1), (h2, w2), ...]
        sampling_locations: (batch_size, num_heads, num_queries, num_levels, num_points, 2)
        attention_weights: (batch_size, num_heads, num_queries, num_levels, num_points)

    Returns:
        output: (batch_size, num_queries, value_dim)
    """
    batch_size, num_heads, num_value, value_dim = value.shape
    num_levels = len(value_shapes)
    num_queries = sampling_locations.shape[2]
    num_points = sampling_locations.shape[4]

    # 验证维度
    assert num_levels == sampling_locations.shape[3] == attention_weights.shape[3], \
        f"尺度数量不匹配: {num_levels} vs {sampling_locations.shape[3]} vs {attention_weights.shape[3]}"

    # 计算每个尺度的 token 数
    split_sizes = [h * w for h, w in value_shapes]
    total_elements = sum(split_sizes)

    if total_elements != num_value:
        print(f"警告: 总元素数不匹配，预期 {num_value}，实际 {total_elements}，正在调整...")
        ratio = num_value / total_elements
        split_sizes = [int(round(s * ratio)) for s in split_sizes]
        split_sizes[-1] += num_value - sum(split_sizes)
        split_sizes = [s for s in split_sizes if s > 0]
        num_levels = len(split_sizes)
        value_shapes = value_shapes[:num_levels]

    # 分割 value
    value_list = jt.split(value, split_sizes, dim=2)

    # 输出初始化
    output = jt.zeros((batch_size, num_queries, value_dim), dtype=value.dtype)

    # 归一化采样位置到 [-1, 1]
    sampling_grids = 2.0 * sampling_locations - 1.0  # [0,1] -> [-1,1]

    head_dim = value_dim // num_heads
    if value_dim % num_heads != 0:
        raise ValueError(f"value_dim({value_dim}) 必须能被 num_heads({num_heads}) 整除")

    for level in range(num_levels):
        h, w = value_shapes[level]
        value_l = value_list[level]  # (B, H, S_l, D)
        if value_l.shape[2] != h * w:
            continue

        # 重塑为 (B, H, h, w, D)
        try:
            value_l = value_l.view(batch_size, num_heads, h, w, value_dim)
        except:
            continue

        # 提取当前尺度的采样网格和权重
        grid_l = sampling_grids[:, :, :, level, :, :]  # (B, H, N, P, 2)
        weight_l = attention_weights[:, :, :, level, :]  # (B, H, N, P)

        # 展平采样点: (B, H, N*P, 2)
        B, H, N, P, _ = grid_l.shape
        grid_l = grid_l.reindex([B, H, N * P, 2], ["i0", "i1", "i2 / @e3", "i3"], extras=[N, P])

        sampled_per_head = []

        for head_idx in range(num_heads):
            # 当前头的特征: (B, h, w, D) -> (B, D, h, w)
            feat = value_l[:, head_idx]  # (B, h, w, D)
            feat = feat.permute(0, 3, 1, 2)  # (B, D, h, w)

            # 当前头的网格: (B, N*P, 2) -> (B, N*P, 1, 2)
            grid_head = grid_l[:, head_idx]  # (B, N*P, 2)
            grid_head = grid_head.reshape(B, N * P, 1, 2)  # (B, H_out, W_out, 2)

            # 网格采样
            try:
                sampled = nn.grid_sample(
                    feat, grid_head,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False
                )  # (B, D, N*P, 1)
                sampled = sampled.squeeze(-1).reshape(B, value_dim, N, P)  # (B, D, N, P)
            except Exception as e:
                jt.log(f"[Error] Level {level}, Head {head_idx}: {e}")
                sampled = jt.zeros((B, value_dim, N, P), dtype=feat.dtype)

            sampled_per_head.append(sampled)

        # 合并头: (H, B, D, N, P) -> (B, H, N, P, D)
        sampled_stack = jt.stack(sampled_per_head, dim=1)  # (B, H, D, N, P)
        sampled_stack = sampled_stack.permute(0, 1, 3, 4, 2)  # (B, H, N, P, D)

        # 应用注意力权重
        weight_l = weight_l.unsqueeze(-1)  # (B, H, N, P, 1)
        weighted = (sampled_stack * weight_l).sum(dim=3)  # (B, H, N, D)

        # 合并头
        weighted = weighted.permute(0, 2, 1, 3).reshape(B, N, -1)  # (B, N, H*D)
        output += weighted

    return output