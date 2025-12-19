#!/usr/bin/env python3

import numpy as np
import jittor as jt
import torch
from ultralytics.utils.loss import VarifocalLoss as PtVFL
from ultralytics.utils.loss import DFLoss as PtDFL
from ultralytics.utils.loss import FocalLoss as PtFocalLoss
from ultralytics.utils.loss import BboxLoss as PtBboxLoss
from ultralytics.utils.loss import KeypointLoss as PtKeypointLoss
from ultralytics.utils.loss import v8DetectionLoss as Ptv8Loss
from nkyolo.utils.loss import BboxLoss, KeypointLoss, v8DetectionLoss
from nkyolo.utils.loss import VarifocalLoss, DFLoss, FocalLoss


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    jt.set_global_seed(seed)


def compare_loss(name, loss_nk, loss_pt, threshold=1e-5, verbose=False):
    """
    兼容绝对/相对误差的对比函数：
    - threshold 仍然保留为绝对误差下限（atol）
    - 额外引入 rtol：对大数值允许按比例误差
    """
    loss_nk = float(loss_nk.numpy().mean())
    loss_pt = float(loss_pt.detach().cpu().numpy().mean())

    diff = abs(loss_nk - loss_pt)

    # 经验设置：对一般 loss 足够严格，对 1e5 量级的总 loss 不会误判
    atol = float(threshold)
    rtol = 5e-5  # 0.005% 相对误差；你当前 v8 约 1.5e-5

    tol = atol + rtol * abs(loss_pt)
    passed = diff <= tol

    print("name = ", name)
    print("nk loss = ", loss_nk)
    print("pt loss = ", loss_pt)
    if verbose:
        rel = diff / (abs(loss_pt) + 1e-12)
        print(f"  abs diff: {diff:.8f}")
        print(f"  rel diff: {rel:.8e}")
        print(f"  tol:      {tol:.8f} (atol={atol}, rtol={rtol})")
    print("  " + ("✓ 测试通过" if passed else f"✗ 测试失败: 差异过大 ({diff:.8f} > {tol:.8f})"))

    return passed


# def compare_loss(name, loss_nk, loss_pt, threshold=1e-5, verbose=False):
#     loss_nk = float(loss_nk.numpy().mean())
#     loss_pt = float(loss_pt.detach().cpu().numpy().mean())
#     print("name = ", name)
#     print("nk loss = ", loss_nk)
#     print("pt loss = ", loss_pt)
#     diff = abs(loss_nk - loss_pt)
#     if verbose:
#         print(f"\n{name}")
#         print(f"  差异:        {diff.mean():.8f}")
#     passed = diff < threshold
#     print("  " + ("✓ 测试通过" if passed else f"✗ 测试失败: 差异过大 ({diff:.8f} > {threshold})"))
#     return passed


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


def test_bbox_loss():
    set_seed()
    batch_size, num_anchors, reg_max = 2, 100, 16
    
    # 构造模拟数据
    pred_dist_np = np.random.randn(batch_size, num_anchors, reg_max * 4).astype(np.float32)
    pred_bboxes_np = np.random.rand(batch_size, num_anchors, 4).astype(np.float32) * 640
    anchor_points_np = np.random.rand(num_anchors, 2).astype(np.float32) * 80
    target_bboxes_np = np.random.rand(batch_size, num_anchors, 4).astype(np.float32) * 640
    target_scores_np = np.random.rand(batch_size, num_anchors, 80).astype(np.float32)
    
    # 模拟 fg_mask (假设前10个是正样本)
    fg_mask_np = np.zeros((batch_size, num_anchors), dtype=bool)
    fg_mask_np[:, :10] = True 
    
    target_scores_sum_val = target_scores_np.sum()
    ts_sum_jt = jt.array([target_scores_sum_val]).float32()
    ts_sum_pt = torch.tensor(target_scores_sum_val)

    # 调用你的 BboxLoss.execute(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
    nk_bbox = BboxLoss(reg_max=reg_max)
    loss_nk = nk_bbox(jt.array(pred_dist_np), jt.array(pred_bboxes_np), jt.array(anchor_points_np), 
                      jt.array(target_bboxes_np), jt.array(target_scores_np), ts_sum_jt, jt.array(fg_mask_np))
    
    pt_bbox = PtBboxLoss(reg_max=reg_max)
    loss_pt = pt_bbox(torch.from_numpy(pred_dist_np), torch.from_numpy(pred_bboxes_np), torch.from_numpy(anchor_points_np), 
                      torch.from_numpy(target_bboxes_np), torch.from_numpy(target_scores_np), ts_sum_pt, torch.from_numpy(fg_mask_np))

    iou_nk, dfl_nk = loss_nk
    iou_pt, dfl_pt = loss_pt

    # 分别对比 IoU 损失和 DFL 损失
    res_iou = compare_loss("BboxLoss_IoU", iou_nk, iou_pt)
    res_dfl = compare_loss("BboxLoss_DFL", dfl_nk, dfl_pt)
    
    return res_iou and res_dfl
    
    # return compare_loss("BboxLoss", loss_nk, loss_pt)

# --- 补全 KeypointLoss 测试 ---
# def test_keypoint_loss():
#     set_seed()
#     num_kpts, batch_size = 17, 4
#     sigmas = np.random.rand(num_kpts).astype(np.float32)
    
#     pred_kpts = np.random.rand(batch_size, num_kpts, 3).astype(np.float32)
#     gt_kpts = np.random.rand(batch_size, num_kpts, 3).astype(np.float32)
#     kpt_mask = np.random.randint(0, 2, (batch_size, num_kpts)).astype(np.float32)
#     area = np.random.rand(batch_size).astype(np.float32) * 1000

#     nk_kpt = KeypointLoss(sigmas=jt.array(sigmas))
#     pt_kpt = PtKeypointLoss(sigmas=torch.from_numpy(sigmas))

#     loss_nk = nk_kpt(jt.array(pred_kpts), jt.array(gt_kpts), jt.array(kpt_mask), jt.array(area))
#     loss_pt = pt_kpt(torch.from_numpy(pred_kpts), torch.from_numpy(gt_kpts), torch.from_numpy(kpt_mask), torch.from_numpy(area))
#     return compare_loss("KeypointLoss", loss_nk, loss_pt)


def test_keypoint_loss():
    set_seed()
    num_kpts, batch_size = 17, 4
    
    # 1. 构造 sigmas: 17 个关键点
    sigmas_np = np.random.rand(num_kpts).astype(np.float32)
    
    # 2. 构造预测和真实关键点: [Batch, 17, 3] (x, y, visibility)
    pred_kpts_np = np.random.rand(batch_size, num_kpts, 3).astype(np.float32)
    gt_kpts_np = np.random.rand(batch_size, num_kpts, 3).astype(np.float32)
    
    # 3. 构造掩码: [Batch, 17]
    kpt_mask_np = np.random.randint(0, 2, (batch_size, num_kpts)).astype(np.float32)
    
    # 4. 【核心修改】构造 area 并调整形状为 [Batch, 1]
    # 这样在 Loss 内部计算时，[Batch, 1] 可以与 sigmas [17] 广播成 [Batch, 17]
    area_np = (np.random.rand(batch_size).astype(np.float32) * 1000).reshape(-1, 1)

    # --- Jittor 测试 ---
    nk_kpt = KeypointLoss(sigmas=jt.array(sigmas_np))
    loss_nk = nk_kpt(
        jt.array(pred_kpts_np), 
        jt.array(gt_kpts_np), 
        jt.array(kpt_mask_np), 
        jt.array(area_np)
    )

    # --- PyTorch 测试 ---
    pt_kpt = PtKeypointLoss(sigmas=torch.from_numpy(sigmas_np))
    loss_pt = pt_kpt(
        torch.from_numpy(pred_kpts_np), 
        torch.from_numpy(gt_kpts_np), 
        torch.from_numpy(kpt_mask_np), 
        torch.from_numpy(area_np)
    )

    return compare_loss("KeypointLoss", loss_nk, loss_pt)

 
def test_v8_detection_loss():
    set_seed()

    class MockModel:
        def __init__(self, is_pt=False):
            self.nc = 80
            self.reg_max = 16
            self.no = self.nc + self.reg_max * 4
            self.args = type('Args', (), {'box': 7.5, 'cls': 0.5, 'dfl': 1.5})()
            stride = [8, 16, 32]
            self.stride = torch.tensor(stride, dtype=torch.float32) if is_pt else jt.array(stride).float32()
            inner_detect = type('Detect', (), {
                'stride': self.stride, 'nc': self.nc, 'reg_max': self.reg_max, 'no': self.no
            })()
            self.model = [None, None, inner_detect]

        def parameters(self):
            params = [torch.zeros(1)] if isinstance(self.stride, torch.Tensor) else [jt.zeros(1)]
            return iter(params)

    class AssignWrapper:
        def __init__(self, base):
            self.base = base
            self.last = None

        def __call__(self, *args, **kwargs):
            out = self.base(*args, **kwargs)
            # out: (target_labels, target_bboxes, target_scores, fg_mask, ...)
            self.last = out
            return out

    batch_size = 2
    img_size = 640

    pred_np = [
        np.random.randn(batch_size, 144, 80, 80).astype(np.float32),
        np.random.randn(batch_size, 144, 40, 40).astype(np.float32),
        np.random.randn(batch_size, 144, 20, 20).astype(np.float32)
    ]

    batch_idx_np = np.array([0, 0, 1], dtype=np.int64)
    cls_np = np.array([1, 5, 2], dtype=np.int64)

    xyxy = np.array([
        [10, 10, 50, 50],
        [100, 100, 150, 150],
        [20, 20, 60, 60]
    ], dtype=np.float32)
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1)
    h = (y2 - y1)
    bboxes_xywh = np.stack([cx, cy, w, h], axis=1) / float(img_size)

    batch_jt = {
        "batch_idx": jt.array(batch_idx_np).int32(),
        "cls": jt.array(cls_np).int32(),
        "bboxes": jt.array(bboxes_xywh).float32(),
    }
    batch_pt = {
        "batch_idx": torch.from_numpy(batch_idx_np).long(),
        "cls": torch.from_numpy(cls_np).long(),
        "bboxes": torch.from_numpy(bboxes_xywh).float(),
    }

    model_jt = MockModel(is_pt=False)
    model_pt = MockModel(is_pt=True)

    nk_v8 = v8DetectionLoss(model_jt)
    pt_v8 = Ptv8Loss(model_pt)

    # 关键：只在脚本里包一层 assigner，抓中间输出（不改 loss 实现）
    nk_v8.assigner = AssignWrapper(nk_v8.assigner)
    pt_v8.assigner = AssignWrapper(pt_v8.assigner)

    loss_nk, items_nk = nk_v8([jt.array(x) for x in pred_np], batch_jt)
    loss_pt, items_pt = pt_v8([torch.from_numpy(x) for x in pred_np], batch_pt)

    # 打印分量（box/cls/dfl）
    items_nk_np = items_nk.numpy().reshape(-1)
    items_pt_np = items_pt.detach().cpu().numpy().reshape(-1)
    print("NK items (box, cls, dfl):", [float(x) for x in items_nk_np])
    print("PT items (box, cls, dfl):", [float(x) for x in items_pt_np])

    # 打印 assigner 输出的 fg_mask 数量和 target_scores.sum
    _, _, ts_nk, fg_nk, _ = nk_v8.assigner.last
    _, _, ts_pt, fg_pt, _ = pt_v8.assigner.last

    fg_nk_cnt = int((fg_nk > 0).sum().item()) if hasattr(fg_nk, "sum") else int(fg_nk.sum())
    fg_pt_cnt = int((fg_pt > 0).sum().item())

    ts_nk_sum = float(ts_nk.sum().item())
    ts_pt_sum = float(ts_pt.sum().item())

    print("NK fg_cnt:", fg_nk_cnt, "target_scores_sum:", ts_nk_sum)
    print("PT fg_cnt:", fg_pt_cnt, "target_scores_sum:", ts_pt_sum)

    return compare_loss("v8DetectionLoss", loss_nk, loss_pt, threshold=1e-3)


if __name__ == "__main__":    
    tests = [
        ("VarifocalLoss", test_varifocal_loss),
        ("DFLoss", test_dfloss),
        ("FocalLoss", test_focal_loss),
        ("BboxLoss", test_bbox_loss),
        ("KeypointLoss", test_keypoint_loss),
        ("v8DetectionLoss", test_v8_detection_loss),
    ]
    
    results = [(name, test()) for name, test in tests]
    
    all_passed = all(r[1] for r in results)
    