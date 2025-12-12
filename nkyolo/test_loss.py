import torch
import torch.nn as nn
import jittor as jt
import numpy as np

# === 导入部分 ===
# 1. 导入 Ultralytics 原版 (作为标准答案)
from ultralytics.utils.loss import VarifocalLoss as PtVFL
from ultralytics.utils.loss import DFLoss as PtDFL
from ultralytics.utils.loss import FocalLoss as PtFL

# 2. 导入你的 Jittor 实现
from nkyolo.utils.loss import VarifocalLoss as JtVFL
from nkyolo.utils.loss import DFLoss as JtDFL
from nkyolo.utils.loss import FocalLoss as JtFL

def check_result(name, pt_val, jt_val):
    """简单的打印对比函数"""
    pt_val = pt_val.item() if isinstance(pt_val, torch.Tensor) else pt_val
    jt_val = jt_val.item() if isinstance(jt_val, jt.Var) else jt_val
    
    diff = abs(pt_val - jt_val)
    status = "✅ 通过" if diff < 1e-5 else "❌ 失败"
    
    print(f"--- {name} ---")
    print(f"输入数据已固定")
    print(f"PyTorch (标准): {pt_val:.6f}")
    print(f"Jittor  (你的): {jt_val:.6f}")
    print(f"误差: {diff:.2e} -> {status}\n")

# ==========================================
# 测试 1: Varifocal Loss
# ==========================================
def test_vfl():
    # 准备数据：2个样本
    # 预测值 (Logits)
    pred_data = [[1.5], [-0.5]] 
    # 真实IoU分数
    gt_score_data = [[0.8], [0.2]]
    # 类别标签
    label_data = [[1.0], [0.0]]

    # PyTorch 运行
    pt_pred = torch.tensor(pred_data, dtype=torch.float32)
    pt_gt = torch.tensor(gt_score_data, dtype=torch.float32)
    pt_lbl = torch.tensor(label_data, dtype=torch.float32)
    
    loss_fcn = PtVFL()
    pt_res = loss_fcn(pt_pred, pt_gt, pt_lbl)

    # Jittor 运行
    jt_pred = jt.array(pred_data).float32()
    jt_gt = jt.array(gt_score_data).float32()
    jt_lbl = jt.array(label_data).float32()
    
    # 注意：你的 execute 返回的是 sum，ultralytics 默认也是 sum 逻辑(取决于实现细节，通常是 mean 后 sum 或直接 sum)
    # 我们这里对比最终标量
    jt_res = JtVFL.execute(jt_pred, jt_gt, jt_lbl)

    check_result("Varifocal Loss", pt_res.sum(), jt_res)

# ==========================================
# 测试 2: DFLoss (Distribution Focal Loss)
# ==========================================
def test_dfl():
    reg_max = 4  # 设小一点方便看数据
    # 预测分布：2个anchor，每个anchor有4个reg_max的概率值
    pred_dist_data = [
        [10.0, 2.0, 1.0, 0.0],  # 倾向于索引0
        [0.0, 1.0, 5.0, 1.0]    # 倾向于索引2
    ]
    # 真实值：连续的坐标值
    target_data = [
        [0.2],  # 靠近0
        [2.8]   # 靠近3
    ]

    # PyTorch 运行
    pt_pred = torch.tensor(pred_dist_data, dtype=torch.float32).view(2, reg_max) # [2, 4]
    pt_target = torch.tensor(target_data, dtype=torch.float32).view(2)           # [2]
    
    loss_fcn = PtDFL(reg_max=reg_max)
    pt_res = loss_fcn(pt_pred, pt_target)

    # Jittor 运行
    jt_pred = jt.array(pred_dist_data).float32() # [2, 4]
    jt_target = jt.array(target_data).float32().view(2)  # [2]
    
    loss_fcn_jt = JtDFL(reg_max=reg_max)
    jt_res = loss_fcn_jt(jt_pred, jt_target)

    check_result("DFLoss", pt_res.mean(), jt_res.mean())

# ==========================================
# 测试 3: Focal Loss
# ==========================================
def test_fl():
    # 模拟二分类
    # 预测值
    pred_data = [[0.8, -0.5], [-1.0, 2.0]]
    # 真实标签
    label_data = [[1.0, 0.0], [0.0, 1.0]]

    # PyTorch 运行
    pt_pred = torch.tensor(pred_data, dtype=torch.float32)
    pt_lbl = torch.tensor(label_data, dtype=torch.float32)
    
    # Ultralytics 的 FocalLoss 需要包装一个 BCE
    loss_fcn = PtFL(nn.BCEWithLogitsLoss(reduction='none'), gamma=1.5, alpha=0.25)
    pt_res = loss_fcn(pt_pred, pt_lbl)

    # Jittor 运行
    jt_pred = jt.array(pred_data).float32()
    jt_lbl = jt.array(label_data).float32()
    
    jt_res = JtFL.execute(jt_pred, jt_lbl, gamma=1.5, alpha=0.25)

    # PyTorch这里我们需要手动做 mean(1).sum() 以对齐你的实现
    pt_final = pt_res.mean(1).sum()
    
    check_result("Focal Loss", pt_final, jt_res)

if __name__ == "__main__":
    print("=== 开始极简数值对比测试 ===\n")
    test_vfl()
    test_dfl()
    test_fl()