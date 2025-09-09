import jittor as jt
import jittor.nn as nn


class FlexibleBatchNorm2d(nn.BatchNorm2d):
    """
    继承Jittor原生BatchNorm2d，支持两种更新模式：
    1. 原生模式（默认）：running_mean/var使用momentum指数移动平均
    2. 无偏模式：running_mean/var = (累计和) / (num_batches_tracked + 1)
    通过use_unbiased_update参数控制，完全兼容原生接口
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, 
                 affine=True, track_running_stats=True, 
                 use_unbiased_update=False):  # 新增控制参数
        # 调用父类初始化（保留原生所有参数）
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )
        
        # 新增功能参数
        self.use_unbiased_update = use_unbiased_update  # 控制是否启用无偏更新
        # 批次计数器：仅在跟踪统计时创建
        if self.track_running_stats:
            self.num_batches_tracked = jt.zeros((), dtype='int64')
        else:
            self.num_batches_tracked = None

    def execute(self, x):
        # 1. 不跟踪统计时，完全沿用原生逻辑
        if not self.track_running_stats:
            return super().execute(x)
        
        # 2. 训练模式：根据use_unbiased_update选择更新方式
        if self.training:
            # 计算当前批次的均值和方差（复用原生逻辑的维度处理）
            batch_mean = x.mean(dim=self.reduction_dims, keepdims=False)
            batch_var = x.var(dim=self.reduction_dims, keepdims=False, unbiased=False)

            # 模式1：无偏更新（除以num_batches_tracked + 1）
            if self.use_unbiased_update:
                total_batches = self.num_batches_tracked + 1
                # 无偏更新公式
                self.running_mean = (self.running_mean * self.num_batches_tracked + batch_mean) / total_batches
                self.running_var = (self.running_var * self.num_batches_tracked + batch_var) / total_batches
                self.num_batches_tracked = total_batches  # 更新计数器
                mean, var = batch_mean, batch_var  # 训练时用当前批次统计
            
            # 模式2：原生更新（指数移动平均）
            else:
                # 直接调用原生更新逻辑（复用父类的running_mean/var更新）
                # 注意：需先备份当前计数器，避免干扰原生逻辑
                backup_counter = self.num_batches_tracked
                self.num_batches_tracked = None  # 临时屏蔽计数器，避免父类误操作
                output = super().execute(x)  # 原生前向传播（含指数移动平均）
                self.num_batches_tracked = backup_counter  # 恢复计数器
                self.num_batches_tracked += 1  # 仅更新计数器，不影响原生统计
                return output  # 直接返回原生计算结果

        # 3. 测试模式：两种模式均使用累计的running_mean/var
        else:
            mean, var = self.running_mean, self.running_var

        # 4. 归一化计算（复用原生逻辑的广播和数值稳定性处理）
        mean = mean.reshape(*self.broadcast_shape)
        var = var.reshape(*self.broadcast_shape)
        x_normalized = (x - mean) / jt.sqrt(var + self.eps)

        # 应用可学习参数（与原生一致）
        if self.affine:
            x_normalized = x_normalized * self.weight.reshape(*self.broadcast_shape) \
                         + self.bias.reshape(*self.broadcast_shape)

        return x_normalized
