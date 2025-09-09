import jittor as jt
import jittor.nn as nn


# ================================
# 多头注意力机制 (MultiheadAttention)
# 功能完全对齐 PyTorch nn.MultiheadAttention，支持参数加载与接口复用
# ================================
class MultiheadAttention(nn.Module):
    """
    Jittor 实现的多头注意力机制，1:1 对齐 PyTorch nn.MultiheadAttention。
    
    关键改进：
    1. 参数命名与 PyTorch 一致（in_proj_weight/in_proj_bias + out_proj子模块）
    2. 支持 key/value 默认为 None（自动复用 query，适配自注意力场景）
    3. 初始化逻辑对齐 PyTorch（Xavier 均匀分布 + 偏置0初始化）
    4. 输入输出格式、掩码处理完全兼容 PyTorch

    参数:
        embed_dim: 输入特征维度（必须能被 num_heads 整除）
        num_heads: 注意力头的数量
        dropout: attention weights 的 dropout 概率，默认为 0.0
        batch_first: 若为 True，输入输出形状为 (batch, seq_len, embed_dim)，否则为 (seq_len, batch, embed_dim)
        bias: 是否在线性投影层使用偏置，默认为 True
        average_attn_weights: 是否对多头的注意力权重取平均，默认为 True
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=False, bias=True, average_attn_weights=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.average_attn_weights = average_attn_weights
        self.bias = bias

        # 校验：嵌入维度必须能被头数整除（多头注意力核心要求）
        if embed_dim % num_heads != 0:
            raise ValueError(f"嵌入维度 {embed_dim} 必须能被头数 {num_heads} 整除")
        self.head_dim = embed_dim // num_heads  # 每个注意力头的维度

        # --------------------------
        # 1. 对齐 PyTorch 参数命名：合并 Q/K/V 投影为 in_proj
        # --------------------------
        # in_proj_weight: 形状 [3*embed_dim, embed_dim]，顺序为 Q→K→V
        self.in_proj_weight = jt.nn.Parameter(jt.empty((3 * embed_dim, embed_dim)))
        # in_proj_bias: 形状 [3*embed_dim]，顺序与 weight 一致（无偏置时注册为空参数）
        if bias:
            self.in_proj_bias = jt.nn.Parameter(jt.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        # 输出投影层：与 PyTorch 命名完全一致（out_proj 子模块，含 weight/bias）
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout 层（仅作用于 attention weights）
        self.dropout_layer = nn.Dropout(dropout)

        # 初始化参数（对齐 PyTorch 官方逻辑）
        self._reset_parameters()

    def _reset_parameters(self):
        """
        对齐 PyTorch 官方参数初始化逻辑：
        - Q/K/V 投影权重：Xavier 均匀分布（保证前向/反向传播梯度稳定）
        - 偏置：全部初始化为 0（避免初始偏置对注意力分布的干扰）
        """
        # 1. 初始化 in_proj_weight（拆分 Q/K/V 三部分分别初始化）
        nn.init.xavier_uniform_(self.in_proj_weight[:self.embed_dim, :])  # Q 投影权重
        nn.init.xavier_uniform_(self.in_proj_weight[self.embed_dim:2*self.embed_dim, :])  # K 投影权重
        nn.init.xavier_uniform_(self.in_proj_weight[2*self.embed_dim:, :])  # V 投影权重

        # 2. 初始化 in_proj_bias（若启用偏置）
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)

        # 3. 初始化 out_proj 子模块（与 PyTorch 逻辑一致）
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def execute(self, query, key=None, value=None, key_padding_mask=None, need_weights=True, attn_mask=None):
        """
        前向传播逻辑，完全对齐 PyTorch 接口：
        - 支持 key/value 默认为 None（自动复用 query，适配自注意力）
        - 支持 batch_first 格式切换
        - 支持 key_padding_mask（序列填充掩码）和 attn_mask（注意力掩码）
        
        参数:
            query: 查询序列，形状 (seq_len, batch, embed_dim) 或 (batch, seq_len, embed_dim)
            key: 键序列，默认 None（复用 query），形状同 query（src_len 可不同）
            value: 值序列，默认 None（复用 key），形状同 key
            key_padding_mask: 键序列填充掩码，形状 (batch, src_len)，True 表示对应位置需掩码
            need_weights: 是否返回注意力权重，默认为 True
            attn_mask: 注意力掩码，形状 (tgt_len, src_len) 或 (batch, tgt_len, src_len)，True 表示对应位置需掩码
        
        返回:
            attn_output: 注意力输出，形状同 query
            attn_weights: 注意力权重（可选），形状 (batch, num_heads, tgt_len, src_len) 或 (batch, tgt_len, src_len)
        """
        # --------------------------
        # 2. 对齐 PyTorch 输入接口：key/value 默认为 None 时复用 query
        # --------------------------
        if key is None:
            key = query
        if value is None:
            value = key

        # --------------------------
        # 3. 处理 batch_first 格式：统一转为 (seq_len, batch, embed_dim) 计算
        # --------------------------
        if self.batch_first:
            query = query.permute(1, 0, 2)  # (batch, tgt_len, dim) → (tgt_len, batch, dim)
            key = key.permute(1, 0, 2)      # (batch, src_len, dim) → (src_len, batch, dim)
            value = value.permute(1, 0, 2)  # 同 key

        # 获取基础维度信息
        tgt_len, batch_size, embed_dim = query.shape  # tgt_len: 查询序列长度
        src_len = key.shape[0]                        # src_len: 键/值序列长度
        if embed_dim != self.embed_dim:
            raise ValueError(f"输入特征维度 {embed_dim} 与模块初始化维度 {self.embed_dim} 不匹配")

        # --------------------------
        # 4. 对齐 PyTorch 投影逻辑：用 in_proj 拆分 Q/K/V
        # --------------------------
        # 拆分 in_proj_weight 为 Q/K/V 单独权重
        q_weight = self.in_proj_weight[:self.embed_dim, :]    # Q 投影权重：[embed_dim, embed_dim]
        k_weight = self.in_proj_weight[self.embed_dim:2*self.embed_dim, :]  # K 投影权重：[embed_dim, embed_dim]
        v_weight = self.in_proj_weight[2*self.embed_dim:, :]  # V 投影权重：[embed_dim, embed_dim]

        # 拆分 in_proj_bias 为 Q/K/V 单独偏置（若启用偏置）
        if self.in_proj_bias is not None:
            q_bias = self.in_proj_bias[:self.embed_dim]        # Q 偏置：[embed_dim]
            k_bias = self.in_proj_bias[self.embed_dim:2*self.embed_dim]  # K 偏置：[embed_dim]
            v_bias = self.in_proj_bias[2*self.embed_dim:]      # V 偏置：[embed_dim]
        else:
            q_bias = k_bias = v_bias = None

        # 执行线性投影（Jittor linear 与 PyTorch 一致：y = x @ W.T + b）
        q = jt.nn.linear(query, q_weight, q_bias)  # Q 投影：(tgt_len, batch, embed_dim)
        k = jt.nn.linear(key, k_weight, k_bias)    # K 投影：(src_len, batch, embed_dim)
        v = jt.nn.linear(value, v_weight, v_bias)  # V 投影：(src_len, batch, embed_dim)

        # --------------------------
        # 5. 多头注意力核心逻辑（与原实现兼容，保证计算正确性）
        # --------------------------
        # 分头：(seq_len, batch, embed_dim) → (batch, num_heads, seq_len, head_dim)
        q = q.view(tgt_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(src_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 3, 0)  # K 转置为 (batch, heads, head_dim, src_len)
        v = v.view(src_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # 计算注意力分数（缩放点积）：score = Q @ K.T / sqrt(head_dim)
        scaling = float(self.head_dim) ** -0.5
        attn_scores = jt.matmul(q, k) * scaling  # (batch, num_heads, tgt_len, src_len)

        # --------------------------
        # 6. 掩码处理（完全对齐 PyTorch 逻辑，支持两种掩码）
        # --------------------------
        # 处理 attn_mask（注意力掩码：屏蔽特定位置的注意力交互）
        if attn_mask is not None:
            # 调整掩码维度：适配 (tgt_len, src_len) 或 (batch, tgt_len, src_len) 输入
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, src_len)
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)              # (batch, 1, tgt_len, src_len)
            else:
                raise ValueError(f"attn_mask 维度必须为 2 或 3，当前为 {attn_mask.ndim}")
            
            # 掩码值替换：将 True 位置的分数设为极小值（避免 Softmax 后被选中）
            min_value = -1e8
            attn_scores = jt.where(attn_mask.bool(), attn_scores, min_value)

        # 处理 key_padding_mask（键填充掩码：屏蔽填充位置的键对所有查询的影响）
        if key_padding_mask is not None:
            # 调整掩码维度：(batch, src_len) → (batch, 1, 1, src_len) → 广播到 (batch, 1, tgt_len, src_len)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            mask = mask.broadcast_shape((batch_size, 1, tgt_len, src_len))
            
            # 掩码值替换（同 attn_mask）
            min_value = -1e8
            attn_scores = jt.where(mask, min_value, attn_scores)

        # 计算注意力权重（Softmax + Dropout）
        attn_weights = nn.softmax(attn_scores, dim=-1)  # 对 src_len 维度归一化
        attn_weights = self.dropout_layer(attn_weights)  # 应用 Dropout

        # 计算注意力输出：output = weight @ V
        attn_output = jt.matmul(attn_weights, v)  # (batch, num_heads, tgt_len, head_dim)

        # --------------------------
        # 7. 合并多头 + 输出投影（对齐 PyTorch 输出格式）
        # --------------------------
        # 合并多头：(batch, num_heads, tgt_len, head_dim) → (tgt_len, batch, embed_dim)
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(tgt_len, batch_size, -1)
        # 输出投影：将合并后的特征映射回 embed_dim 维度
        attn_output = self.out_proj(attn_output)

        # 恢复 batch_first 格式（若输入为 batch_first）
        if self.batch_first:
            attn_output = attn_output.permute(1, 0, 2)  # (tgt_len, batch, dim) → (batch, tgt_len, dim)

        # --------------------------
        # 8. 注意力权重处理（按 average_attn_weights 决定是否平均多头）
        # --------------------------
        if need_weights:
            # 若需平均多头权重：(batch, num_heads, tgt_len, src_len) → (batch, tgt_len, src_len)
            if self.average_attn_weights and self.num_heads > 1:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output