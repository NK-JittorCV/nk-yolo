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
