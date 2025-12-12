import torch
import torch.nn as nn
import jittor as jt
import numpy as np

# 1. 导入你的 Jittor 实现
from nkyolo.utils.loss import VarifocalLoss as JtVFL
from nkyolo.utils.loss import DFLoss as JtDFL
from nkyolo.utils.loss import FocalLoss as JtFL

# 2. 导入 Ultralytics 组件 (跳过 FocalLoss，我们在下面手动实现)
from ultralytics.utils.loss import VarifocalLoss as PtVFL
from ultralytics.utils.loss import DFLoss as PtDFL

# --- 手动定义标准的 PyTorch Focal Loss 以避开库版本冲突 ---
class ManualPyTorchFocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, label):
        loss = self.bce(pred, label)
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if self.alpha > 0:
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        # 对齐你的实现：对 dim=1 求均值，对 batch 求和
        return loss.mean(1).sum()

def check_result(name, pt_val, jt_val):
    pt_val = pt_val.item() if isinstance(pt_val, torch.Tensor) else pt_val
    jt_val = jt_val.item() if isinstance(jt_val, jt.Var) else jt_val
    
    # 转换为 float 以避免 numpy 打印格式问题
    pt_val = float(pt_val)
    jt_val = float(jt_val)

    diff = abs(pt_val - jt_val)
    status = "✅ 通过" if diff < 1e-4 else "❌ 失败"
    
    print(f"--- {name} ---")
    print(f"PyTorch (标准): {pt_val:.6f}")
    print(f"Jittor  (你的): {jt_val:.6f}")
    print(f"误差: {diff:.2e} -> {status}\n")

def test_vfl():
    # 预测值 (Logits)
    pred_data = [[1.5], [-0.5]] 
    # 真实IoU分数
    gt_score_data = [[0.8], [0.2]]
    # 类别标签
    label_data = [[1.0], [0.0]]

    # PyTorch
    pt_pred = torch.tensor(pred_data, dtype=torch.float32)
    pt_gt = torch.tensor(gt_score_data, dtype=torch.float32)
    pt_lbl = torch.tensor(label_data, dtype=torch.float32)
    loss_fcn = PtVFL()
    pt_res = loss_fcn(pt_pred, pt_gt, pt_lbl)

    # Jittor
    jt_pred = jt.array(pred_data).float32()
    jt_gt = jt.array(gt_score_data).float32()
    jt_lbl = jt.array(label_data).float32()
    jt_res = JtVFL.execute(jt_pred, jt_gt, jt_lbl)

    check_result("Varifocal Loss", pt_res.sum(), jt_res)

def test_dfl():
    reg_max = 4
    # 预测分布
    pred_dist_data = [
        [10.0, 2.0, 1.0, 0.0],
        [0.0, 1.0, 5.0, 1.0]
    ]
    # 真实值
    target_data = [[0.2], [2.8]]

    # PyTorch
    pt_pred = torch.tensor(pred_dist_data, dtype=torch.float32).view(2, reg_max)
    pt_target = torch.tensor(target_data, dtype=torch.float32).view(2)
    loss_fcn = PtDFL(reg_max=reg_max)
    pt_res = loss_fcn(pt_pred, pt_target)

    # Jittor
    jt_pred = jt.array(pred_dist_data).float32()
    jt_target = jt.array(target_data).float32().view(2)
    loss_fcn_jt = JtDFL(reg_max=reg_max)
    jt_res = loss_fcn_jt(jt_pred, jt_target)

    check_result("DFLoss", pt_res.mean(), jt_res.mean())

def test_fl():
    # 预测值
    pred_data = [[0.8, -0.5], [-1.0, 2.0]]
    # 真实标签
    label_data = [[1.0, 0.0], [0.0, 1.0]]

    # PyTorch (使用上面定义的手动类)
    pt_pred = torch.tensor(pred_data, dtype=torch.float32)
    pt_lbl = torch.tensor(label_data, dtype=torch.float32)
    
    loss_fcn = ManualPyTorchFocalLoss(gamma=1.5, alpha=0.25)
    pt_res = loss_fcn(pt_pred, pt_lbl)

    # Jittor
    jt_pred = jt.array(pred_data).float32()
    jt_lbl = jt.array(label_data).float32()
    
    jt_res = JtFL.execute(jt_pred, jt_lbl, gamma=1.5, alpha=0.25)

    check_result("Focal Loss", pt_res, jt_res)

if __name__ == "__main__":
    print("=== 开始极简数值对比测试 (v3) ===\n")
    test_vfl()
    test_dfl()
    test_fl()