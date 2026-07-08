import jittor as jt
import torch
import numpy as np

def manual_bce_jt(pred, label):
    return jt.maximum(pred, 0.0) - pred * label + jt.log(1.0 + jt.exp(-jt.abs(pred)))

def test_bce_behavior():
    print("====== BCE behavior difference diagnosis ======")
    
    B, C = 2, 1000
    pred_np = np.random.randn(B, C).astype(np.float32)
    label_np = np.random.rand(B, C).astype(np.float32) # Soft labels

    pt_pred = torch.from_numpy(pred_np)
    pt_label = torch.from_numpy(label_np)
    pt_bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    pt_loss = pt_bce(pt_pred, pt_label).sum()
    print(f"PyTorch (Sum): {pt_loss.item():.4f}")

    jt_pred = jt.array(pred_np)
    jt_label = jt.array(label_np)
    jt_bce_layer = jt.nn.BCEWithLogitsLoss() 
    jt_loss_default = jt_bce_layer(jt_pred, jt_label).sum()
    print(f"Jittor (Default Class): {jt_loss_default.item():.4f} <--- root cause (too small)")

    jt_loss_manual = manual_bce_jt(jt_pred, jt_label).sum()
    print(f"Jittor (Manual Fix):    {jt_loss_manual.item():.4f} <--- expected value")

    diff = abs(pt_loss.item() - jt_loss_manual.item())
    if diff < 1e-3:
        print("\n✅ Conclusion: must use manual BCE in Seg/Pose loss instead of self.bce")
    else:
        print("\n❌ Conclusion: manual implementation still differs, check the math")

if __name__ == "__main__":
    test_bce_behavior()
