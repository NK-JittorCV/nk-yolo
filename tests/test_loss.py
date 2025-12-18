#!/usr/bin/env python3

import numpy as np
import jittor as jt
import torch
from ultralytics.utils.loss import VarifocalLoss as PtVFL
from ultralytics.utils.loss import DFLoss as PtDFL
from ultralytics.utils.loss import FocalLoss as PtFocalLoss

from nkyolo.utils.loss import VarifocalLoss, DFLoss, FocalLoss


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    jt.set_global_seed(seed)


def compare_loss(name, loss_nk, loss_pt, threshold=1e-5, verbose=False):
    loss_nk = float(loss_nk.numpy().mean())
    loss_pt = float(loss_pt.detach().cpu().numpy().mean())
    print("name = ", name)
    print("nk loss = ", loss_nk)
    print("pt loss = ", loss_pt)
    diff = abs(loss_nk - loss_pt)
    if verbose:
        print(f"\n{name}")
        print(f"  差异:        {diff.mean():.8f}")
    passed = diff < threshold
    print("  " + ("✓ 测试通过" if passed else f"✗ 测试失败: 差异过大 ({diff:.8f} > {threshold})"))
    return passed


def test_varifocal_loss():
    set_seed()
    
    batch_size, num_anchors, num_classes = 2, 100, 80
    pred_score_np = np.random.randn(batch_size, num_anchors, num_classes).astype(np.float32)
    gt_score_np = np.random.rand(batch_size, num_anchors, num_classes).astype(np.float32)
    label_np = np.random.randint(0, 2, (batch_size, num_anchors, num_classes)).astype(np.float32)
    
    pred_score_jt = jt.array(pred_score_np)
    gt_score_jt = jt.array(gt_score_np)
    label_jt = jt.array(label_np)
    
    pred_score_pt = torch.from_numpy(pred_score_np)
    gt_score_pt = torch.from_numpy(gt_score_np)
    label_pt = torch.from_numpy(label_np)
    
    nk_vfl = VarifocalLoss()
    pt_vfl = PtVFL()
    
    loss_nk = nk_vfl(pred_score_jt, gt_score_jt, label_jt)
    loss_pt = pt_vfl(pred_score_pt, gt_score_pt, label_pt)
    return compare_loss("VarifocalLoss", loss_nk, loss_pt)


def test_dfloss():
    set_seed()
    
    # DFLoss 需要 (N, reg_max) 和 (N,) 的输入
    # 根据实际使用，pred_dist 是 (N, reg_max)，target 是 (N,)
    n_samples, reg_max = 200, 16
    
    pred_dist_np = np.random.randn(n_samples, reg_max).astype(np.float32)
    target_np = np.random.rand(n_samples).astype(np.float32) * (reg_max - 1)
    
    pred_dist_jt = jt.array(pred_dist_np)
    target_jt = jt.array(target_np)
    pred_dist_pt = torch.from_numpy(pred_dist_np)
    target_pt = torch.from_numpy(target_np)
    
    nk_dfl = DFLoss(reg_max=16)
    pt_dfl = PtDFL(reg_max=16)
    
    loss_nk = nk_dfl(pred_dist_jt, target_jt)
    loss_pt = pt_dfl(pred_dist_pt, target_pt)
    return compare_loss("DFLoss", loss_nk, loss_pt)


def test_focal_loss():
    set_seed()
    
    batch_size, num_anchors, num_classes = 2, 100, 80
    pred_np = np.random.randn(batch_size, num_anchors, num_classes).astype(np.float32)
    label_np = np.random.randint(0, 2, (batch_size, num_anchors, num_classes)).astype(np.float32)
    
    pred_jt = jt.array(pred_np)
    label_jt = jt.array(label_np)
    pred_pt = torch.from_numpy(pred_np)
    label_pt = torch.from_numpy(label_np)
    
    nk_focal = FocalLoss()
    pt_focal = PtFocalLoss()
    
    loss_nk = nk_focal(pred_jt, label_jt)
    loss_pt = pt_focal(pred_pt, label_pt)    
    return compare_loss("FocalLoss", loss_nk, loss_pt)


if __name__ == "__main__":    
    tests = [
        ("VarifocalLoss", test_varifocal_loss),
        ("DFLoss", test_dfloss),
        ("FocalLoss", test_focal_loss),
    ]
    
    results = [(name, test()) for name, test in tests]
    
    all_passed = all(r[1] for r in results)
    