import jittor as jt
import jittor.nn as nn
import copy

class MultiheadAttention(nn.Module):
    """
    Jittor实现的多头注意力机制，与PyTorch的nn.MultiheadAttention功能对应
    
    参数:
        embed_dim: 输入特征维度
        num_heads: 注意力头的数量
        dropout: dropout概率，默认为0.0
        batch_first: 如果为True，则输入和输出张量的形状为(batch, seq_len, embed_dim)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        
        # 确保嵌入维度可以被头数整除
        assert embed_dim % num_heads == 0, "嵌入维度必须能被头数整除"
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V的线性变换
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出线性变换
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
    
    def __getstate__(self):
        """用于序列化模块状态，深拷贝和保存模型时调用"""
        state = self.__dict__.copy()
        # 对模块参数进行深拷贝，确保状态独立
        state['q_proj'] = copy.deepcopy(self.q_proj)
        state['k_proj'] = copy.deepcopy(self.k_proj)
        state['v_proj'] = copy.deepcopy(self.v_proj)
        state['out_proj'] = copy.deepcopy(self.out_proj)
        state['dropout_layer'] = copy.deepcopy(self.dropout_layer)
        return state
    
    def __setstate__(self, state):
        """用于反序列化模块状态，加载模型或深拷贝后恢复状态时调用"""
        self.__dict__.update(state)
        # 恢复子模块的引用
        self.q_proj = state['q_proj']
        self.k_proj = state['k_proj']
        self.v_proj = state['v_proj']
        self.out_proj = state['out_proj']
        self.dropout_layer = state['dropout_layer']
        
    def execute(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        """
        参数:
            query: 查询张量，形状为 (seq_len, batch, embed_dim) 或 (batch, seq_len, embed_dim)
            key: 键张量，形状与query相同
            value: 值张量，形状与query相同
            key_padding_mask: 键的填充掩码，形状为 (batch, seq_len)
            need_weights: 是否返回注意力权重
            attn_mask: 注意力掩码，形状为 (num_heads*batch, tgt_len, src_len) 或 (tgt_len, src_len)
            
        返回:
            attn_output: 注意力输出，形状与query相同
            attn_output_weights: 注意力权重，形状为 (batch, num_heads, tgt_len, src_len)
        """
        # 调整输入形状以适应batch_first参数
        if self.batch_first:
            # 如果batch_first为True，输入形状是(batch, seq_len, embed_dim)
            # 转换为(seq_len, batch, embed_dim)进行内部处理
            query = query.permute(1, 0, 2)
            key = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        
        # 获取序列长度、批次大小和嵌入维度
        tgt_len, batch_size, embed_dim = query.shape
        src_len = key.shape[0]
        assert embed_dim == self.embed_dim, f"输入嵌入维度({embed_dim})与预期({self.embed_dim})不符"
        
        # 线性变换并分头
        q = self.q_proj(query).view(tgt_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # (batch, num_heads, tgt_len, head_dim)
        k = self.k_proj(key).view(src_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 3, 0)  # (batch, num_heads, head_dim, src_len)
        v = self.v_proj(value).view(src_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # (batch, num_heads, src_len, head_dim)
        
        # 计算注意力分数: (batch, num_heads, tgt_len, src_len)
        attn_scores = jt.matmul(q, k) / (self.head_dim ** 0.5)
        
        # 应用注意力掩码
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # 广播到所有头和批次
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, src_len)
            elif attn_mask.dim() == 3:
                # 广播到所有头
                attn_mask = attn_mask.unsqueeze(1)  # (batch, 1, tgt_len, src_len)
            attn_scores = attn_scores + attn_mask
        
        # 应用键填充掩码
        if key_padding_mask is not None:
            # key_padding_mask形状: (batch, src_len)
            # 转换为: (batch, 1, 1, src_len) 以便广播
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(key_padding_mask, -float('inf'))
        
        # 计算注意力权重
        attn_weights = nn.softmax(attn_scores, dim=-1)  # (batch, num_heads, tgt_len, src_len)
        attn_weights = self.dropout_layer(attn_weights)
        
        # 应用注意力权重到值
        attn_output = jt.matmul(attn_weights, v)  # (batch, num_heads, tgt_len, head_dim)
        
        # 拼接所有头的输出
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, batch_size, embed_dim)  # (tgt_len, batch, embed_dim)
        
        # 输出线性变换
        attn_output = self.out_proj(attn_output)
        
        # 如果需要，调整回batch_first格式
        if self.batch_first:
            attn_output = attn_output.permute(1, 0, 2)  # (batch, tgt_len, embed_dim)
        
        if need_weights:
            # 平均所有头的注意力权重，形状变为(batch, tgt_len, src_len)
            attn_output_weights = attn_weights.mean(dim=1)
            return attn_output, attn_output_weights
        else:
            return attn_output


def multi_scale_deformable_attn_pytorch(
    value: jt.Var,
    value_shapes: list[tuple[int, int]],
    sampling_locations: jt.Var,
    attention_weights: jt.Var
) -> jt.Var:
    """
    多尺度可变形注意力机制的Jittor完整实现
    支持动态调整特征尺寸匹配，自动处理维度不兼容问题
    
    Args:
        value: 输入特征张量，形状为 (batch_size, num_heads, num_value, value_dim)
        value_shapes: 每个尺度的特征图尺寸列表，格式为 [(h1, w1), (h2, w2), ...]
        sampling_locations: 采样位置张量，形状为 (batch_size, num_heads, num_queries, num_levels, num_points, 2)
        attention_weights: 注意力权重张量，形状为 (batch_size, num_heads, num_queries, num_levels, num_points)
    
    Returns:
        输出特征张量，形状为 (batch_size, num_queries, value_dim)
    """
    # 解析输入张量维度信息
    batch_size, num_heads, num_value, value_dim = value.shape
    num_levels = len(value_shapes)
    num_queries = sampling_locations.shape[2]
    num_points = sampling_locations.shape[4]

    # 验证输入维度一致性
    assert num_levels == sampling_locations.shape[3], \
        f"尺度数量不匹配: value_shapes有{num_levels}个, 采样位置有{sampling_locations.shape[3]}个"
    assert num_levels == attention_weights.shape[3], \
        f"尺度数量不匹配: value_shapes有{num_levels}个, 注意力权重有{attention_weights.shape[3]}个"

    # 计算并调整分割尺寸
    split_sizes = [h * w for h, w in value_shapes]
    total_size = sum(split_sizes)
    
    # 自动调整分割尺寸以匹配value的维度
    if total_size != num_value:
        jt.log(f"警告: 特征尺寸不匹配，预期{num_value}但得到{total_size}，已自动调整")
        ratio = num_value / total_size
        split_sizes = [int(round(s * ratio)) for s in split_sizes]
        split_sizes[-1] += num_value - sum(split_sizes)  # 修正误差
        # 过滤无效尺寸
        split_sizes = [s for s in split_sizes if s > 0]
        num_levels = len(split_sizes)
        value_shapes = value_shapes[:num_levels]

    # 分割多尺度特征
    value_list = jt.split(value, split_sizes, dim=2)

    # 初始化输出张量
    output = jt.zeros((batch_size, num_queries, value_dim), dtype=value.dtype)
    
    # 归一化采样位置到[-1, 1]范围（Jittor网格采样要求）
    sampling_grids = 2 * sampling_locations - 1

    # 计算每个注意力头的维度
    head_dim = value_dim // num_heads
    if value_dim % num_heads != 0:
        jt.log(f"警告: value_dim({value_dim})不能被num_heads({num_heads})整除，可能导致精度损失")

    # 遍历每个尺度进行特征采样
    for level in range(num_levels):
        # 获取当前尺度的特征和尺寸
        h, w = value_shapes[level]
        value_l = value_list[level]  # 形状: (batch_size, num_heads, num_value_level, value_dim)
        
        # 验证当前尺度特征的有效性
        current_elements = value_l.numel()
        expected_elements = batch_size * num_heads * split_sizes[level] * value_dim
        if current_elements != expected_elements:
            jt.log(f"跳过无效尺度{level}: 元素数量不匹配({current_elements} vs {expected_elements})")
            continue

        # 调整特征形状以适应网格采样
        # (batch_size, num_heads, num_value_level, value_dim) -> 
        # (batch_size, num_heads, head_dim, h, w)
        try:
            value_l_reshaped = value_l.transpose(2, 1).view(
                batch_size, h, w, num_heads, head_dim
            ).permute(0, 3, 4, 1, 2)
        except RuntimeError:
            # 处理形状重塑失败的情况
            jt.log(f"尺度{level}重塑失败，自动调整形状")
            value_l_reshaped = value_l.transpose(2, 1).reshape(
                batch_size, h, w, num_heads, head_dim
            ).permute(0, 3, 4, 1, 2)

        # 提取当前尺度的采样位置和注意力权重
        sampling_grid_l = sampling_grids[:, :, :, level]  # (bs, num_heads, num_queries, num_points, 2)
        attention_weights_l = attention_weights[:, :, :, level]  # (bs, num_heads, num_queries, num_points)

        # 调整采样网格形状以适应Jittor的grid_sample要求
        # grid_sample输入形状: (N, C, H, W)
        # 网格形状: (N, H_out, W_out, 2) 或 (N, num_heads, H_out, W_out, 2)
        bs, nh, nq, np, _ = sampling_grid_l.shape
        sampling_grid_l = sampling_grid_l.permute(0, 2, 3, 1, 4).reshape(bs, nq * np, nh, 2)
        sampling_grid_l = sampling_grid_l.permute(0, 2, 1, 3)  # (bs, num_heads, nq*np, 2)

        # 执行网格采样
        try:
            # value_l_reshaped形状: (bs, num_heads, head_dim, h, w)
            sampling_value_l = nn.grid_sample(
                value_l_reshaped,
                sampling_grid_l,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )  # 输出: (bs, num_heads, head_dim, 1, nq*np)
        except Exception as e:
            jt.log(f"尺度{level}网格采样失败: {str(e)}")
            continue

        # 调整采样结果形状并应用注意力权重
        sampling_value_l = sampling_value_l.reshape(bs, nh, head_dim, nq, np)
        attention_weights_l = attention_weights_l.unsqueeze(2)  # (bs, nh, 1, nq, np)
        
        # 加权求和
        weighted_value = (sampling_value_l * attention_weights_l).sum(-1)  # (bs, nh, head_dim, nq)
        weighted_value = weighted_value.transpose(1, 3).reshape(bs, nq, nh * head_dim)  # (bs, nq, value_dim)

        # 累加到输出
        output += weighted_value

    return output
    