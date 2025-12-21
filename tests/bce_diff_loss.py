import jittor as jt
import torch
import numpy as np

def manual_bce_jt(pred, label):
    # 手动实现 Element-wise BCE
    return jt.maximum(pred, 0.0) - pred * label + jt.log(1.0 + jt.exp(-jt.abs(pred)))

def test_bce_behavior():
    print("====== BCE 行为差异诊断 ======")
    
    B, C = 2, 1000
    pred_np = np.random.randn(B, C).astype(np.float32)
    label_np = np.random.rand(B, C).astype(np.float32) # Soft labels

    # 1. PyTorch (reduction='none') -> Sum
    pt_pred = torch.from_numpy(pred_np)
    pt_label = torch.from_numpy(label_np)
    pt_bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    pt_loss = pt_bce(pt_pred, pt_label).sum()
    print(f"PyTorch (Sum): {pt_loss.item():.4f}")

    # 2. Jittor (self.bce 默认行为)
    jt_pred = jt.array(pred_np)
    jt_label = jt.array(label_np)
    jt_bce_layer = jt.nn.BCEWithLogitsLoss() # 默认 reduction='mean'
    jt_loss_default = jt_bce_layer(jt_pred, jt_label).sum()
    # 注意：这里虽然调用了 sum()，但 jt_bce_layer 内部已经做了一次 mean，所以是对一个标量求和
    print(f"Jittor (Default Class): {jt_loss_default.item():.4f} <--- 错误根源 (太小)")

    # 3. Jittor (手动 Element-wise)
    jt_loss_manual = manual_bce_jt(jt_pred, jt_label).sum()
    print(f"Jittor (Manual Fix):    {jt_loss_manual.item():.4f} <--- 期望值")

    diff = abs(pt_loss.item() - jt_loss_manual.item())
    if diff < 1e-3:
        print("\n✅ 诊断结论: 必须在 Seg/Pose Loss 中使用手动 BCE 替代 self.bce")
    else:
        print("\n❌ 诊断结论: 手动实现仍有差异，需检查数学公式")

if __name__ == "__main__":
    test_bce_behavior()