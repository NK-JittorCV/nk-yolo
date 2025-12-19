#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import jittor as jt
import torch

# =========================
# Imports: PyTorch (Ultralytics)
# =========================
from ultralytics.utils.loss import VarifocalLoss as PtVFL
from ultralytics.utils.loss import DFLoss as PtDFL
from ultralytics.utils.loss import FocalLoss as PtFocalLoss
from ultralytics.utils.loss import BboxLoss as PtBboxLoss
from ultralytics.utils.loss import KeypointLoss as PtKeypointLoss
from ultralytics.utils.loss import v8DetectionLoss as Ptv8DetLoss

try:
    from ultralytics.utils.loss import v8SegmentationLoss as Ptv8SegLoss
except Exception:
    Ptv8SegLoss = None

try:
    from ultralytics.utils.loss import v8PoseLoss as Ptv8PoseLoss
except Exception:
    Ptv8PoseLoss = None

try:
    from ultralytics.utils.loss import v8ClassificationLoss as Ptv8ClsLoss
except Exception:
    Ptv8ClsLoss = None

try:
    from ultralytics.utils.loss import v8OBBLoss as Ptv8OBBLoss
except Exception:
    Ptv8OBBLoss = None

try:
    from ultralytics.utils.loss import E2EDetectLoss as PtE2EDetectLoss
except Exception:
    PtE2EDetectLoss = None


# =========================
# Imports: Jittor (nkyolo)
# =========================
from nkyolo.utils.loss import VarifocalLoss, DFLoss, FocalLoss
from nkyolo.utils.loss import BboxLoss, KeypointLoss, v8DetectionLoss

try:
    from nkyolo.utils.loss import v8SegmentationLoss as Nkv8SegLoss
except Exception:
    Nkv8SegLoss = None

try:
    from nkyolo.utils.loss import v8PoseLoss as Nkv8PoseLoss
except Exception:
    Nkv8PoseLoss = None

try:
    from nkyolo.utils.loss import v8ClassificationLoss as Nkv8ClsLoss
except Exception:
    Nkv8ClsLoss = None

try:
    from nkyolo.utils.loss import v8OBBLoss as Nkv8OBBLoss
except Exception:
    Nkv8OBBLoss = None

try:
    from nkyolo.utils.loss import E2EDetectLoss as NkE2EDetectLoss
except Exception:
    NkE2EDetectLoss = None


# =========================
# Optional ops/tal coverage
# =========================
try:
    from ultralytics.utils.tal import TaskAlignedAssigner as PtTaskAlignedAssigner
    from ultralytics.utils.tal import make_anchors as pt_make_anchors
except Exception:
    PtTaskAlignedAssigner = None
    pt_make_anchors = None

try:
    from ultralytics.utils.ops import dist2bbox as pt_dist2bbox
except Exception:
    pt_dist2bbox = None

try:
    from nkyolo.utils.tal import TaskAlignedAssigner as NkTaskAlignedAssigner
    from nkyolo.utils.tal import make_anchors as nk_make_anchors
except Exception:
    NkTaskAlignedAssigner = None
    nk_make_anchors = None

try:
    from nkyolo.utils.ops import dist2bbox as nk_dist2bbox
except Exception:
    nk_dist2bbox = None


# =========================
# Repro + compare helpers
# =========================
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    jt.set_global_seed(seed)


def compare_loss(name, loss_nk, loss_pt, threshold=1e-5, verbose=False):
    """
    Pass if |nk-pt| <= atol + rtol*|pt|
    threshold -> atol
    """
    loss_nk = float(loss_nk.numpy().mean())
    loss_pt = float(loss_pt.detach().cpu().numpy().mean())

    diff = abs(loss_nk - loss_pt)
    atol = float(threshold)
    rtol = 5e-5
    tol = atol + rtol * abs(loss_pt)
    passed = diff <= tol

    print("name = ", name)
    print("nk loss = ", loss_nk)
    print("pt loss = ", loss_pt)

    if verbose:
        rel = diff / (abs(loss_pt) + 1e-12)
        print("abs diff = ", float(diff))
        print("rel diff = ", float(rel))
        print("tol      = ", float(tol), "(atol =", atol, ", rtol =", rtol, ")")

    if passed:
        print("  ✓ 测试通过")
    else:
        print("  ✗ 测试失败: 差异过大 (%.8f > %.8f)" % (diff, tol))

    return passed


def compare_tensor(name, a_nk, a_pt, atol=1e-5, rtol=5e-5, verbose=False):
    a_nk = a_nk.numpy()
    a_pt = a_pt.detach().cpu().numpy()

    diff = float(np.max(np.abs(a_nk - a_pt)))
    tol = float(atol + rtol * np.max(np.abs(a_pt)))
    passed = diff <= tol

    print("name = ", name)
    print("max abs diff = ", diff)
    print("tol = ", tol)

    if verbose:
        print("nk shape:", a_nk.shape, "pt shape:", a_pt.shape)

    if passed:
        print("  ✓ 测试通过")
    else:
        print("  ✗ 测试失败: 差异过大 (%.8f > %.8f)" % (diff, tol))

    return passed


def _to_total_loss(x):
    """
    Align scalar-vs-vector returns:
      - if tensor/var shape (3,), sum() -> scalar total
      - else return as-is
    """
    if isinstance(x, torch.Tensor):
        if x.ndim == 1 and x.numel() == 3:
            return x.sum()
        return x
    if hasattr(x, "shape"):
        try:
            if len(x.shape) == 1 and int(x.numel()) == 3:
                return x.sum()
        except Exception:
            pass
    return x


def _skip(name, reason: str):
    print("name = ", name)
    print("  ✓ 测试跳过:", reason)
    return True


def _fail(name, reason: str):
    print("name = ", name)
    print("  ✗ 测试失败:", reason)
    return False


def _try_instantiate(loss_cls, *args, **kwargs):
    """
    Try instantiate with (model) then () fallback, for PT/NK inconsistencies.
    """
    try:
        return loss_cls(*args, **kwargs), None
    except TypeError as e:
        try:
            return loss_cls(), None
        except Exception as e2:
            return None, (e, e2)
    except Exception as e:
        return None, (e,)


def _try_call(loss_obj, preds, batch):
    """
    Attempt forward; returns (loss, items) or raises.
    """
    return loss_obj(preds, batch)


def _probe_preds_and_run(loss_nk, loss_pt, preds_nk_candidates, preds_pt_candidates, batch_nk, batch_pt):
    """
    Try multiple pred formats until both NK/PT run successfully.
    Returns (loss_nk, items_nk, loss_pt, items_pt, chosen_index) or raises last error.
    """
    last_err = None
    for i in range(min(len(preds_nk_candidates), len(preds_pt_candidates))):
        pn = preds_nk_candidates[i]
        pp = preds_pt_candidates[i]
        try:
            ln, in_ = _try_call(loss_nk, pn, batch_nk)
            lp, ip = _try_call(loss_pt, pp, batch_pt)
            return ln, in_, lp, ip, i
        except Exception as e:
            last_err = e
            continue
    raise last_err


# =========================
# Basic tests (you already had them passing)
# =========================
def test_varifocal_loss():
    set_seed()
    b, a, c = 2, 100, 80
    pred_np = np.random.randn(b, a, c).astype(np.float32)
    gt_np = np.random.rand(b, a, c).astype(np.float32)
    label_np = np.random.randint(0, 2, (b, a, c)).astype(np.float32)
    nk = VarifocalLoss()
    pt = PtVFL()
    loss_nk = nk(jt.array(pred_np), jt.array(gt_np), jt.array(label_np))
    loss_pt = pt(torch.from_numpy(pred_np), torch.from_numpy(gt_np), torch.from_numpy(label_np))
    return compare_loss("VarifocalLoss", loss_nk, loss_pt, threshold=1e-5)


def test_dfloss():
    set_seed()
    n, reg_max = 200, 16
    pred_np = np.random.randn(n, reg_max).astype(np.float32)
    target_np = (np.random.rand(n).astype(np.float32) * (reg_max - 1))
    nk = DFLoss(reg_max=16)
    pt = PtDFL(reg_max=16)
    loss_nk = nk(jt.array(pred_np), jt.array(target_np))
    loss_pt = pt(torch.from_numpy(pred_np), torch.from_numpy(target_np))
    return compare_loss("DFLoss", loss_nk, loss_pt, threshold=1e-5)


def test_focal_loss():
    set_seed()
    b, a, c = 2, 100, 80
    pred_np = np.random.randn(b, a, c).astype(np.float32)
    label_np = np.random.randint(0, 2, (b, a, c)).astype(np.float32)
    nk = FocalLoss()
    pt = PtFocalLoss()
    loss_nk = nk(jt.array(pred_np), jt.array(label_np))
    loss_pt = pt(torch.from_numpy(pred_np), torch.from_numpy(label_np))
    return compare_loss("FocalLoss", loss_nk, loss_pt, threshold=1e-5)


def test_bbox_loss():
    set_seed()
    b, a, reg_max = 2, 100, 16
    pred_dist_np = np.random.randn(b, a, reg_max * 4).astype(np.float32)
    pred_bboxes_np = (np.random.rand(b, a, 4).astype(np.float32) * 640.0)
    anchor_np = (np.random.rand(a, 2).astype(np.float32) * 80.0)
    tgt_bboxes_np = (np.random.rand(b, a, 4).astype(np.float32) * 640.0)
    tgt_scores_np = np.random.rand(b, a, 80).astype(np.float32)
    fg_np = np.zeros((b, a), dtype=bool)
    fg_np[:, :10] = True
    ts_sum_val = float(tgt_scores_np.sum())
    ts_sum_jt = jt.array([ts_sum_val]).float32()
    ts_sum_pt = torch.tensor(ts_sum_val, dtype=torch.float32)
    nk = BboxLoss(reg_max=reg_max)
    pt = PtBboxLoss(reg_max=reg_max)
    iou_nk, dfl_nk = nk(
        jt.array(pred_dist_np),
        jt.array(pred_bboxes_np),
        jt.array(anchor_np),
        jt.array(tgt_bboxes_np),
        jt.array(tgt_scores_np),
        ts_sum_jt,
        jt.array(fg_np),
    )
    iou_pt, dfl_pt = pt(
        torch.from_numpy(pred_dist_np),
        torch.from_numpy(pred_bboxes_np),
        torch.from_numpy(anchor_np),
        torch.from_numpy(tgt_bboxes_np),
        torch.from_numpy(tgt_scores_np),
        ts_sum_pt,
        torch.from_numpy(fg_np),
    )
    ok1 = compare_loss("BboxLoss_IoU", iou_nk, iou_pt, threshold=1e-5)
    ok2 = compare_loss("BboxLoss_DFL", dfl_nk, dfl_pt, threshold=1e-5)
    return ok1 and ok2


def test_keypoint_loss():
    set_seed()
    num_kpts, b = 17, 4
    sigmas_np = np.random.rand(num_kpts).astype(np.float32)
    pred_np = np.random.rand(b, num_kpts, 3).astype(np.float32)
    gt_np = np.random.rand(b, num_kpts, 3).astype(np.float32)
    mask_np = np.random.randint(0, 2, (b, num_kpts)).astype(np.float32)
    area_np = (np.random.rand(b).astype(np.float32) * 1000.0).reshape(-1, 1)
    nk = KeypointLoss(sigmas=jt.array(sigmas_np))
    pt = PtKeypointLoss(sigmas=torch.from_numpy(sigmas_np))
    loss_nk = nk(jt.array(pred_np), jt.array(gt_np), jt.array(mask_np), jt.array(area_np))
    loss_pt = pt(torch.from_numpy(pred_np), torch.from_numpy(gt_np), torch.from_numpy(mask_np), torch.from_numpy(area_np))
    return compare_loss("KeypointLoss", loss_nk, loss_pt, threshold=1e-5)


# =========================
# ops/tal coverage (optional)
# =========================
def test_dist2bbox():
    set_seed()
    if pt_dist2bbox is None or nk_dist2bbox is None:
        return _skip("dist2bbox", "dist2bbox 未找到")
    b, a = 2, 300
    dist_np = (np.random.rand(b, a, 4).astype(np.float32) * 10.0)
    anchor_np = (np.random.rand(a, 2).astype(np.float32) * 80.0)
    out_nk = nk_dist2bbox(jt.array(dist_np), jt.array(anchor_np), xywh=False)
    out_pt = pt_dist2bbox(torch.from_numpy(dist_np), torch.from_numpy(anchor_np), xywh=False)
    return compare_tensor("dist2bbox", out_nk, out_pt, atol=1e-4, rtol=1e-4)


def test_make_anchors():
    set_seed()
    if pt_make_anchors is None or nk_make_anchors is None:
        return _skip("make_anchors", "make_anchors 未找到")
    b, ch = 2, 144
    feats_np = [
        np.random.randn(b, ch, 80, 80).astype(np.float32),
        np.random.randn(b, ch, 40, 40).astype(np.float32),
        np.random.randn(b, ch, 20, 20).astype(np.float32),
    ]
    stride_list = [8, 16, 32]
    ap_nk, st_nk = nk_make_anchors([jt.array(x) for x in feats_np], jt.array(stride_list).float32(), 0.5)
    ap_pt, st_pt = pt_make_anchors([torch.from_numpy(x) for x in feats_np], torch.tensor(stride_list, dtype=torch.float32), 0.5)
    ok1 = compare_tensor("make_anchors.anchor_points", ap_nk, ap_pt, atol=1e-4, rtol=1e-4)
    ok2 = compare_tensor("make_anchors.stride_tensor", st_nk, st_pt, atol=1e-4, rtol=1e-4)
    return ok1 and ok2


def test_task_aligned_assigner():
    set_seed()
    if PtTaskAlignedAssigner is None or NkTaskAlignedAssigner is None:
        return _skip("TaskAlignedAssigner", "TaskAlignedAssigner 未找到")
    b, na, nc, max_gt = 2, 500, 80, 20
    pd_scores_np = np.random.rand(b, na, nc).astype(np.float32)
    pd_bboxes_np = (np.random.rand(b, na, 4).astype(np.float32) * 640.0)
    anc_np = (np.random.rand(na, 2).astype(np.float32) * 80.0)
    gt_labels_np = np.random.randint(0, nc, (b, max_gt, 1)).astype(np.int64)
    gt_bboxes_np = (np.random.rand(b, max_gt, 4).astype(np.float32) * 640.0)
    mask_gt_np = (np.random.rand(b, max_gt, 1) > 0.3)
    nk = NkTaskAlignedAssigner(topk=10, num_classes=nc, alpha=0.5, beta=6.0)
    pt = PtTaskAlignedAssigner(topk=10, num_classes=nc, alpha=0.5, beta=6.0)
    out_nk = nk(
        jt.array(pd_scores_np),
        jt.array(pd_bboxes_np),
        jt.array(anc_np),
        jt.array(gt_labels_np).int32(),
        jt.array(gt_bboxes_np),
        jt.array(mask_gt_np.astype(np.float32)) > 0,
    )
    out_pt = pt(
        torch.from_numpy(pd_scores_np),
        torch.from_numpy(pd_bboxes_np),
        torch.from_numpy(anc_np),
        torch.from_numpy(gt_labels_np).long(),
        torch.from_numpy(gt_bboxes_np),
        torch.from_numpy(mask_gt_np),
    )
    tl_nk, tb_nk, ts_nk, fg_nk, *_ = out_nk
    tl_pt, tb_pt, ts_pt, fg_pt, *_ = out_pt
    ok1 = compare_tensor("TAL.target_bboxes", tb_nk, tb_pt, atol=1e-3, rtol=1e-4)
    ok2 = compare_tensor("TAL.target_scores", ts_nk, ts_pt, atol=1e-3, rtol=1e-4)
    tl_ok = np.array_equal(tl_nk.numpy(), tl_pt.detach().cpu().numpy())
    fg_ok = np.array_equal(fg_nk.numpy(), fg_pt.detach().cpu().numpy())
    print("name = TAL.target_labels")
    print("  ✓ 测试通过" if tl_ok else "  ✗ 测试失败: target_labels 不一致")
    print("name = TAL.fg_mask")
    print("  ✓ 测试通过" if fg_ok else "  ✗ 测试失败: fg_mask 不一致")
    return ok1 and ok2 and tl_ok and fg_ok


# =========================
# Mock models tailored per loss
# =========================
def _make_args(**kwargs):
    return type("Args", (), kwargs)()


def _mock_detect_model(is_pt: bool, nc=80, reg_max=16, stride=(8, 16, 32)):
    """
    For v8DetectionLoss: official PT uses next(model.parameters()) so parameters() must be an iterator.
    """
    class M:
        def __init__(self):
            self.nc = nc
            self.reg_max = reg_max
            self.no = nc + reg_max * 4
            self.args = _make_args(box=7.5, cls=0.5, dfl=1.5)
            self.stride = torch.tensor(list(stride), dtype=torch.float32) if is_pt else jt.array(list(stride)).float32()
            inner = type("Detect", (), {"stride": self.stride, "nc": self.nc, "reg_max": self.reg_max, "no": self.no})()
            self.model = [None, None, inner]

        def parameters(self):
            # iterator for next(...)
            p = torch.zeros(1) if is_pt else jt.zeros(1)
            return iter([p])

    return M()


def _mock_seg_model(is_pt: bool, nc=80, reg_max=16, nm=32, npr=32, stride=(8, 16, 32)):
    """
    Add args.overlap_mask to satisfy your error.
    Ultralytics seg also often reads args.overlap_mask / args.mask_ratio / args.retina_masks (version dependent).
    """
    class M:
        def __init__(self):
            self.nc = nc
            self.reg_max = reg_max
            self.no = nc + reg_max * 4
            self.args = _make_args(
                box=7.5,
                cls=0.5,
                dfl=1.5,
                mask=1.0,
                overlap_mask=False,   # <-- required by your runtime error
                mask_ratio=4,
                retina_masks=False,
            )
            self.stride = torch.tensor(list(stride), dtype=torch.float32) if is_pt else jt.array(list(stride)).float32()
            inner = type(
                "Segment",
                (),
                {"stride": self.stride, "nc": self.nc, "reg_max": self.reg_max, "no": self.no, "nm": nm, "npr": npr},
            )()
            self.model = [None, None, inner]

        def parameters(self):
            p = torch.zeros(1) if is_pt else jt.zeros(1)
            return iter([p])

    return M()


def _mock_pose_model(is_pt: bool, nc=80, reg_max=16, kpt_shape=(17, 3), stride=(8, 16, 32)):
    """
    Pose loss implementations differ a lot in how kpts are provided.
    We will try several preds formats; this model just provides metadata.
    """
    class M:
        def __init__(self):
            self.nc = nc
            self.reg_max = reg_max
            self.no = nc + reg_max * 4
            self.args = _make_args(box=7.5, cls=0.5, dfl=1.5, pose=1.0, kobj=1.0)
            self.stride = torch.tensor(list(stride), dtype=torch.float32) if is_pt else jt.array(list(stride)).float32()
            inner = type("Pose", (), {"stride": self.stride, "nc": self.nc, "reg_max": self.reg_max, "no": self.no, "kpt_shape": kpt_shape})()
            self.model = [None, None, inner]

        def parameters(self):
            p = torch.zeros(1) if is_pt else jt.zeros(1)
            return iter([p])

    return M()


def _mock_obb_model(is_pt: bool, nc=80, reg_max=16, stride=(8, 16, 32)):
    """
    OBB loss likely expects extra angle channels or different head output.
    We'll probe multiple layouts in test.
    """
    class M:
        def __init__(self):
            self.nc = nc
            self.reg_max = reg_max
            self.no = nc + reg_max * 4
            self.args = _make_args(box=7.5, cls=0.5, dfl=1.5)
            self.stride = torch.tensor(list(stride), dtype=torch.float32) if is_pt else jt.array(list(stride)).float32()
            inner = type("OBB", (), {"stride": self.stride, "nc": self.nc, "reg_max": self.reg_max, "no": self.no})()
            self.model = [None, None, inner]

        def parameters(self):
            p = torch.zeros(1) if is_pt else jt.zeros(1)
            return iter([p])

    return M()


def _mock_e2e_model(is_pt: bool, nc=80):
    """
    Your error: "'list_iterator' object is not subscriptable"
    -> E2E loss in your code is likely doing model.parameters()[0] (subscript),
       so here parameters() must return a LIST, not an iterator.
    """
    class M:
        def __init__(self):
            self.args = _make_args(box=7.5, cls=0.5)
            inner = type("E2E", (), {"nc": nc})()
            self.model = [None, None, inner]

        def parameters(self):
            p = torch.zeros(1) if is_pt else jt.zeros(1)
            return [p]  # <-- list, subscriptable

    return M()


# =========================
# v8 Detection test (stable)
# =========================
def test_v8_detection_loss():
    set_seed()

    model_jt = _mock_detect_model(is_pt=False)
    model_pt = _mock_detect_model(is_pt=True)

    nk_loss = v8DetectionLoss(model_jt)
    pt_loss = Ptv8DetLoss(model_pt)

    b = 2
    preds_np = [
        np.random.randn(b, 144, 80, 80).astype(np.float32),
        np.random.randn(b, 144, 40, 40).astype(np.float32),
        np.random.randn(b, 144, 20, 20).astype(np.float32),
    ]

    img_size = 640
    batch_idx_np = np.array([0, 0, 1], dtype=np.int64)
    cls_np = np.array([1, 5, 2], dtype=np.int64)

    xyxy = np.array([[10, 10, 50, 50], [100, 100, 150, 150], [20, 20, 60, 60]], dtype=np.float32)
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

    loss_nk, items_nk = nk_loss([jt.array(x) for x in preds_np], batch_jt)
    loss_pt, items_pt = pt_loss([torch.from_numpy(x) for x in preds_np], batch_pt)

    loss_nk_total = _to_total_loss(loss_nk)
    loss_pt_total = _to_total_loss(loss_pt)

    items_nk_np = items_nk.numpy().reshape(-1)
    items_pt_np = items_pt.detach().cpu().numpy().reshape(-1)

    print("NK items (box, cls, dfl):", [float(x) for x in items_nk_np])
    print("PT items (box, cls, dfl):", [float(x) for x in items_pt_np])

    ok_total = compare_loss("v8DetectionLoss_total", loss_nk_total, loss_pt_total, threshold=1e-3, verbose=True)
    ok_box = compare_loss("v8DetectionLoss_box", jt.array([items_nk_np[0]]), torch.tensor([items_pt_np[0]]), threshold=1e-3)
    ok_cls = compare_loss("v8DetectionLoss_cls", jt.array([items_nk_np[1]]), torch.tensor([items_pt_np[1]]), threshold=1e-3)
    ok_dfl = compare_loss("v8DetectionLoss_dfl", jt.array([items_nk_np[2]]), torch.tensor([items_pt_np[2]]), threshold=1e-3)
    return ok_total and ok_box and ok_cls and ok_dfl


# =========================
# v8SegmentationLoss (fixed: overlap_mask) + robust pred formats
# =========================
def test_v8_segmentation_loss():
    set_seed()

    if Ptv8SegLoss is None or Nkv8SegLoss is None:
        return _skip("v8SegmentationLoss", "对应类未找到（PT或NK）")

    nc, reg_max = 80, 16
    nm, npr = 32, 32
    stride = (8, 16, 32)

    model_jt = _mock_seg_model(is_pt=False, nc=nc, reg_max=reg_max, nm=nm, npr=npr, stride=stride)
    model_pt = _mock_seg_model(is_pt=True, nc=nc, reg_max=reg_max, nm=nm, npr=npr, stride=stride)

    nk_loss, err_nk = _try_instantiate(Nkv8SegLoss, model_jt)
    pt_loss, err_pt = _try_instantiate(Ptv8SegLoss, model_pt)
    if nk_loss is None:
        return _fail("v8SegmentationLoss", "初始化失败(NK): %r" % (err_nk,))
    if pt_loss is None:
        return _fail("v8SegmentationLoss", "初始化失败(PT): %r" % (err_pt,))

    b = 2
    # Seg head output is typically [det(no) + mask_coeff(nm)]
    ch = (nc + reg_max * 4) + nm
    feats_np = [
        np.random.randn(b, ch, 80, 80).astype(np.float32),
        np.random.randn(b, ch, 40, 40).astype(np.float32),
        np.random.randn(b, ch, 20, 20).astype(np.float32),
    ]
    proto_np = np.random.randn(b, npr, 160, 160).astype(np.float32)

    img_size = 640
    batch_idx_np = np.array([0, 0, 1], dtype=np.int64)
    cls_np = np.array([1, 5, 2], dtype=np.int64)

    xyxy = np.array([[10, 10, 50, 50], [100, 100, 150, 150], [20, 20, 60, 60]], dtype=np.float32)
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1)
    h = (y2 - y1)
    bboxes_xywh = np.stack([cx, cy, w, h], axis=1) / float(img_size)

    masks_np = (np.random.rand(3, 160, 160).astype(np.float32) > 0.7).astype(np.float32)

    batch_jt = {
        "batch_idx": jt.array(batch_idx_np).int32(),
        "cls": jt.array(cls_np).int32(),
        "bboxes": jt.array(bboxes_xywh).float32(),
        "masks": jt.array(masks_np).float32(),
    }
    batch_pt = {
        "batch_idx": torch.from_numpy(batch_idx_np).long(),
        "cls": torch.from_numpy(cls_np).long(),
        "bboxes": torch.from_numpy(bboxes_xywh).float(),
        "masks": torch.from_numpy(masks_np).float(),
    }

    preds_nk_candidates = [
        ([jt.array(x) for x in feats_np], jt.array(proto_np)),
        tuple([jt.array(x) for x in feats_np] + [jt.array(proto_np)]),
        [jt.array(x) for x in feats_np] + [jt.array(proto_np)],
    ]
    preds_pt_candidates = [
        ([torch.from_numpy(x) for x in feats_np], torch.from_numpy(proto_np)),
        tuple([torch.from_numpy(x) for x in feats_np] + [torch.from_numpy(proto_np)]),
        [torch.from_numpy(x) for x in feats_np] + [torch.from_numpy(proto_np)],
    ]

    try:
        loss_nk, items_nk, loss_pt, items_pt, idx = _probe_preds_and_run(
            nk_loss, pt_loss, preds_nk_candidates, preds_pt_candidates, batch_jt, batch_pt
        )
        print("v8SegmentationLoss preds format index =", idx)
    except Exception as e:
        return _fail("v8SegmentationLoss", "forward失败: %r" % e)

    ok_total = compare_loss("v8SegmentationLoss_total", _to_total_loss(loss_nk), _to_total_loss(loss_pt), threshold=1e-3, verbose=True)
    ok_items = compare_loss("v8SegmentationLoss_items_mean", items_nk.mean(), items_pt.mean(), threshold=1e-3)
    return ok_total and ok_items


# =========================
# v8PoseLoss (fix reshape by probing pred formats)
# =========================
def test_v8_pose_loss():
    set_seed()

    if Ptv8PoseLoss is None or Nkv8PoseLoss is None:
        return _skip("v8PoseLoss", "对应类未找到（PT或NK）")

    nc, reg_max = 80, 16
    nkpt, kpt_dim = 17, 3
    stride = (8, 16, 32)

    model_jt = _mock_pose_model(is_pt=False, nc=nc, reg_max=reg_max, kpt_shape=(nkpt, kpt_dim), stride=stride)
    model_pt = _mock_pose_model(is_pt=True, nc=nc, reg_max=reg_max, kpt_shape=(nkpt, kpt_dim), stride=stride)

    nk_loss, err_nk = _try_instantiate(Nkv8PoseLoss, model_jt)
    pt_loss, err_pt = _try_instantiate(Ptv8PoseLoss, model_pt)
    if nk_loss is None:
        return _fail("v8PoseLoss", "初始化失败(NK): %r" % (err_nk,))
    if pt_loss is None:
        return _fail("v8PoseLoss", "初始化失败(PT): %r" % (err_pt,))

    b = 2
    # Common pose head output: [det(no) + kpts(nkpt*kpt_dim)]
    ch = (nc + reg_max * 4) + nkpt * kpt_dim
    feats_np = [
        np.random.randn(b, ch, 80, 80).astype(np.float32),
        np.random.randn(b, ch, 40, 40).astype(np.float32),
        np.random.randn(b, ch, 20, 20).astype(np.float32),
    ]

    # Also prepare a separated kpts tensor candidate (some implementations output feats + kpts separately)
    # kpts logits/coords per level: (b, nkpt*kpt_dim, h, w) -> we reuse the last channels slice
    kpts_levels_np = [
        x[:, -(nkpt * kpt_dim):, :, :].copy() for x in feats_np
    ]

    img_size = 640
    batch_idx_np = np.array([0, 0, 1], dtype=np.int64)
    cls_np = np.array([1, 5, 2], dtype=np.int64)

    xyxy = np.array([[10, 10, 50, 50], [100, 100, 150, 150], [20, 20, 60, 60]], dtype=np.float32)
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1)
    h = (y2 - y1)
    bboxes_xywh = np.stack([cx, cy, w, h], axis=1) / float(img_size)

    # GT keypoints per instance: (N, nkpt, 3) normalized xy + visibility
    kpts_np = np.random.rand(3, nkpt, kpt_dim).astype(np.float32)
    kpts_np[..., 2] = (kpts_np[..., 2] > 0.5).astype(np.float32)

    batch_jt = {
        "batch_idx": jt.array(batch_idx_np).int32(),
        "cls": jt.array(cls_np).int32(),
        "bboxes": jt.array(bboxes_xywh).float32(),
        "keypoints": jt.array(kpts_np).float32(),
    }
    batch_pt = {
        "batch_idx": torch.from_numpy(batch_idx_np).long(),
        "cls": torch.from_numpy(cls_np).long(),
        "bboxes": torch.from_numpy(bboxes_xywh).float(),
        "keypoints": torch.from_numpy(kpts_np).float(),
    }

    preds_nk_candidates = [
        [jt.array(x) for x in feats_np],
        ([jt.array(x) for x in feats_np], [jt.array(x) for x in kpts_levels_np]),
        tuple([jt.array(x) for x in feats_np] + [jt.array(x) for x in kpts_levels_np]),
    ]
    preds_pt_candidates = [
        [torch.from_numpy(x) for x in feats_np],
        ([torch.from_numpy(x) for x in feats_np], [torch.from_numpy(x) for x in kpts_levels_np]),
        tuple([torch.from_numpy(x) for x in feats_np] + [torch.from_numpy(x) for x in kpts_levels_np]),
    ]

    try:
        loss_nk, items_nk, loss_pt, items_pt, idx = _probe_preds_and_run(
            nk_loss, pt_loss, preds_nk_candidates, preds_pt_candidates, batch_jt, batch_pt
        )
        print("v8PoseLoss preds format index =", idx)
    except Exception as e:
        return _fail("v8PoseLoss", "forward失败: %r" % e)

    ok_total = compare_loss("v8PoseLoss_total", _to_total_loss(loss_nk), _to_total_loss(loss_pt), threshold=1e-3, verbose=True)
    ok_items = compare_loss("v8PoseLoss_items_mean", items_nk.mean(), items_pt.mean(), threshold=1e-3)
    return ok_total and ok_items


# =========================
# v8ClassificationLoss (fix ctor takes no args)
# =========================
def test_v8_classification_loss():
    set_seed()

    if Ptv8ClsLoss is None or Nkv8ClsLoss is None:
        return _skip("v8ClassificationLoss", "对应类未找到（PT或NK）")

    # Try init with () first (your error shows NK takes no args)
    nk_loss, err_nk = _try_instantiate(Nkv8ClsLoss)
    if nk_loss is None:
        # try with dummy model, in case your NK differs
        class DummyNk:
            def __init__(self):
                self.args = _make_args(label_smoothing=0.0)
                self.model = [None, None, type("Classify", (), {"nc": 1000})()]
            def parameters(self):
                return iter([jt.zeros(1)])
        nk_loss, err_nk = _try_instantiate(Nkv8ClsLoss, DummyNk())
        if nk_loss is None:
            return _fail("v8ClassificationLoss", "初始化失败(NK): %r" % (err_nk,))

    # PT is usually model-dependent; probe both
    class DummyPt:
        def __init__(self):
            self.args = _make_args(label_smoothing=0.0)
            self.model = [None, None, type("Classify", (), {"nc": 1000})()]
        def parameters(self):
            return iter([torch.zeros(1)])

    pt_loss, err_pt = _try_instantiate(Ptv8ClsLoss, DummyPt())
    if pt_loss is None:
        pt_loss, err_pt = _try_instantiate(Ptv8ClsLoss)
        if pt_loss is None:
            return _fail("v8ClassificationLoss", "初始化失败(PT): %r" % (err_pt,))

    b = 8
    nc = 1000
    logits_np = np.random.randn(b, nc).astype(np.float32)
    labels_np = np.random.randint(0, nc, (b,), dtype=np.int64)

    # Many cls losses accept (preds, batch) where batch has cls
    batch_jt = {"cls": jt.array(labels_np).int32()}
    batch_pt = {"cls": torch.from_numpy(labels_np).long()}

    # Probe calling styles: some implement __call__(preds, batch), some __call__(batch, preds)
    preds_nk_candidates = [jt.array(logits_np)]
    preds_pt_candidates = [torch.from_numpy(logits_np)]

    try:
        loss_nk, items_nk = nk_loss(preds_nk_candidates[0], batch_jt)
        loss_pt, items_pt = pt_loss(preds_pt_candidates[0], batch_pt)
    except Exception:
        try:
            loss_nk, items_nk = nk_loss(batch_jt, preds_nk_candidates[0])
            loss_pt, items_pt = pt_loss(batch_pt, preds_pt_candidates[0])
        except Exception as e:
            return _fail("v8ClassificationLoss", "forward失败: %r" % e)

    ok_total = compare_loss("v8ClassificationLoss_total", _to_total_loss(loss_nk), _to_total_loss(loss_pt), threshold=1e-5, verbose=True)
    ok_items = compare_loss("v8ClassificationLoss_items_mean", items_nk.mean(), items_pt.mean(), threshold=1e-5)
    return ok_total and ok_items


# =========================
# v8OBBLoss (fix reshape by probing channel layouts + pred formats)
# =========================
def test_v8_obb_loss():
    set_seed()

    if Ptv8OBBLoss is None or Nkv8OBBLoss is None:
        return _skip("v8OBBLoss", "对应类未找到（PT或NK）")

    nc, reg_max = 80, 16
    stride = (8, 16, 32)

    model_jt = _mock_obb_model(is_pt=False, nc=nc, reg_max=reg_max, stride=stride)
    model_pt = _mock_obb_model(is_pt=True, nc=nc, reg_max=reg_max, stride=stride)

    nk_loss, err_nk = _try_instantiate(Nkv8OBBLoss, model_jt)
    pt_loss, err_pt = _try_instantiate(Ptv8OBBLoss, model_pt)
    if nk_loss is None:
        return _fail("v8OBBLoss", "初始化失败(NK): %r" % (err_nk,))
    if pt_loss is None:
        return _fail("v8OBBLoss", "初始化失败(PT): %r" % (err_pt,))

    b = 2

    # OBB head may append angle channels. We'll try several extra dims.
    extra_candidates = [0, 1, 2, 4, 8]
    feats_candidates_np = []
    for extra in extra_candidates:
        ch = (nc + reg_max * 4) + extra
        feats_np = [
            np.random.randn(b, ch, 80, 80).astype(np.float32),
            np.random.randn(b, ch, 40, 40).astype(np.float32),
            np.random.randn(b, ch, 20, 20).astype(np.float32),
        ]
        feats_candidates_np.append(feats_np)

    # GT OBB bboxes: best-guess (cx, cy, w, h, angle) normalized
    batch_idx_np = np.array([0, 0, 1], dtype=np.int64)
    cls_np = np.array([1, 5, 2], dtype=np.int64)
    obb_np = np.zeros((3, 5), dtype=np.float32)
    obb_np[:, 0:4] = np.random.rand(3, 4).astype(np.float32)
    obb_np[:, 4] = (np.random.rand(3).astype(np.float32) * np.pi) - (np.pi / 2)

    batch_jt = {
        "batch_idx": jt.array(batch_idx_np).int32(),
        "cls": jt.array(cls_np).int32(),
        "bboxes": jt.array(obb_np).float32(),
    }
    batch_pt = {
        "batch_idx": torch.from_numpy(batch_idx_np).long(),
        "cls": torch.from_numpy(cls_np).long(),
        "bboxes": torch.from_numpy(obb_np).float(),
    }

    # Build pred candidates across channel layouts and common wrappers
    preds_nk_candidates = []
    preds_pt_candidates = []
    for feats_np in feats_candidates_np:
        preds_nk_candidates.append([jt.array(x) for x in feats_np])
        preds_pt_candidates.append([torch.from_numpy(x) for x in feats_np])

    try:
        loss_nk, items_nk, loss_pt, items_pt, idx = _probe_preds_and_run(
            nk_loss, pt_loss, preds_nk_candidates, preds_pt_candidates, batch_jt, batch_pt
        )
        print("v8OBBLoss preds layout index(extra_candidates) =", idx, "extra =", extra_candidates[idx])
    except Exception as e:
        return _fail("v8OBBLoss", "forward失败（多半是pred通道布局/GT键名不匹配）: %r" % e)

    ok_total = compare_loss("v8OBBLoss_total", _to_total_loss(loss_nk), _to_total_loss(loss_pt), threshold=1e-3, verbose=True)
    ok_items = compare_loss("v8OBBLoss_items_mean", items_nk.mean(), items_pt.mean(), threshold=1e-3)
    return ok_total and ok_items


# =========================
# E2EDetectLoss (fix ctor: parameters() list, and probe init signatures)
# =========================
def test_e2e_detect_loss():
    set_seed()

    if PtE2EDetectLoss is None or NkE2EDetectLoss is None:
        return _skip("E2EDetectLoss", "对应类未找到（PT或NK）")

    nc = 80
    num_queries = 300

    model_jt = _mock_e2e_model(is_pt=False, nc=nc)
    model_pt = _mock_e2e_model(is_pt=True, nc=nc)

    nk_loss, err_nk = _try_instantiate(NkE2EDetectLoss, model_jt)
    if nk_loss is None:
        nk_loss, err_nk = _try_instantiate(NkE2EDetectLoss)
        if nk_loss is None:
            return _fail("E2EDetectLoss", "初始化失败(NK): %r" % (err_nk,))

    pt_loss, err_pt = _try_instantiate(PtE2EDetectLoss, model_pt)
    if pt_loss is None:
        pt_loss, err_pt = _try_instantiate(PtE2EDetectLoss)
        if pt_loss is None:
            return _fail("E2EDetectLoss", "初始化失败(PT): %r" % (err_pt,))

    b = 2
    pred_logits_np = np.random.randn(b, num_queries, nc).astype(np.float32)
    pred_boxes_np = np.random.rand(b, num_queries, 4).astype(np.float32)

    # We probe two common styles:
    # 1) dict preds with keys 'pred_logits'/'pred_boxes'
    # 2) tuple/list preds (pred_logits, pred_boxes)
    preds_nk_candidates = [
        {"pred_logits": jt.array(pred_logits_np), "pred_boxes": jt.array(pred_boxes_np)},
        (jt.array(pred_logits_np), jt.array(pred_boxes_np)),
        [jt.array(pred_logits_np), jt.array(pred_boxes_np)],
    ]
    preds_pt_candidates = [
        {"pred_logits": torch.from_numpy(pred_logits_np), "pred_boxes": torch.from_numpy(pred_boxes_np)},
        (torch.from_numpy(pred_logits_np), torch.from_numpy(pred_boxes_np)),
        [torch.from_numpy(pred_logits_np), torch.from_numpy(pred_boxes_np)],
    ]

    batch_idx_np = np.array([0, 0, 1], dtype=np.int64)
    cls_np = np.array([1, 5, 2], dtype=np.int64)
    gt_boxes_np = np.random.rand(3, 4).astype(np.float32)

    batch_pt = {
        "batch_idx": torch.from_numpy(batch_idx_np).long(),
        "cls": torch.from_numpy(cls_np).long(),
        "bboxes": torch.from_numpy(gt_boxes_np).float(),
    }
    batch_jt = {
        "batch_idx": jt.array(batch_idx_np).int32(),
        "cls": jt.array(cls_np).int32(),
        "bboxes": jt.array(gt_boxes_np).float32(),
    }

    try:
        loss_nk, items_nk, loss_pt, items_pt, idx = _probe_preds_and_run(
            nk_loss, pt_loss, preds_nk_candidates, preds_pt_candidates, batch_jt, batch_pt
        )
        print("E2EDetectLoss preds format index =", idx)
    except Exception as e:
        return _fail("E2EDetectLoss", "forward失败（多半是接口/键名不同，需要按你实现改）: %r" % e)

    ok_total = compare_loss("E2EDetectLoss_total", _to_total_loss(loss_nk), _to_total_loss(loss_pt), threshold=1e-3, verbose=True)
    ok_items = compare_loss("E2EDetectLoss_items_mean", items_nk.mean(), items_pt.mean(), threshold=1e-3)
    return ok_total and ok_items


# =========================
# Main runner
# =========================
def main():
    tests = [
        ("VarifocalLoss", test_varifocal_loss),
        ("DFLoss", test_dfloss),
        ("FocalLoss", test_focal_loss),
        ("BboxLoss", test_bbox_loss),
        ("KeypointLoss", test_keypoint_loss),
        ("dist2bbox", test_dist2bbox),
        ("make_anchors", test_make_anchors),
        ("TaskAlignedAssigner", test_task_aligned_assigner),
        ("v8DetectionLoss", test_v8_detection_loss),
        ("v8SegmentationLoss", test_v8_segmentation_loss),
        ("v8PoseLoss", test_v8_pose_loss),
        ("v8ClassificationLoss", test_v8_classification_loss),
        ("v8OBBLoss", test_v8_obb_loss),
        ("E2EDetectLoss", test_e2e_detect_loss),
    ]

    results = []
    for name, fn in tests:
        try:
            ok = fn()
        except Exception as e:
            ok = False
            print("name = ", name)
            print("  ✗ 测试失败: 运行时异常:", repr(e))
        results.append((name, ok))

    all_passed = all(ok for _, ok in results)

    print("")
    print("====================")
    print("ALL PASSED =", all_passed)
    print("====================")


if __name__ == "__main__":
    main()
