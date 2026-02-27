# NK-YOLO 🚀, AGPL-3.0 License
# Refer to https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py

import jittor as jt
import jittor.nn as nn

from nkyolo.utils import LOGGER
from nkyolo.utils.metrics import OKS_SIGMA
from nkyolo.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from nkyolo.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors

from .metrics import bbox_iou, probiou
from .tal import bbox2dist

def fix_manual_bce_with_logits(logits, labels):
    return jt.maximum(logits, 0.0) - logits * labels + jt.log(1.0 + jt.exp(-jt.abs(logits)))


class VarifocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def execute(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        prob = pred_score.sigmoid()
        weight = alpha * prob.pow(gamma) * (1 - label) + gt_score * label
        loss_ele = fix_manual_bce_with_logits(pred_score.float(), gt_score.float())
        loss_weighted = loss_ele * weight
        return loss_weighted.mean(1).sum()


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def execute(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = fix_manual_bce_with_logits(pred, label)
        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
        nn.cross_entropy_loss(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + nn.cross_entropy_loss(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def execute(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = jt.Var(0.0)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def execute(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = jt.Var(0.0)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def execute(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (jt.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula

        sigmas_sq = (2 * self.sigmas).pow(2).view(1, -1)
        area_val = (area + 1e-9).view(-1, 1)

        # e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        e = d / (sigmas_sq * area_val * 2)
        return (kpt_loss_factor.view(-1, 1) * ((1 - jt.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        params = model.parameters()  
        if params:
            device = "cuda" if jt.flags.use_cuda else "cpu"
        else:
            device = "cpu"
        self.device = device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss()
        # self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        # m.stride is already jt.Var (set in tasks.py DetectionModel.__init__ at line 403)
        # All sources set stride as jt.Var: tasks.py lines 403, 409, 535, 1110, 1322
        self.stride = m.stride  # Direct assignment - m.stride is always jt.Var from source
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max)
        self.proj = jt.arange(m.reg_max, dtype=jt.float32)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = jt.zeros((batch_size, 0, ne - 1), dtype=jt.float32)
        else:
            i: jt.Var = targets[:, 0]  # image index
            # Critical fix: Always cast to int32 regardless of current dtype
            # This handles the case where model precision switches between epochs:
            # - Epoch 1: float16 mode -> bboxes is float16 -> targets[:, 0] is float16
            # - Epoch 2+: After validation, model is float32 -> bboxes is float32 -> targets[:, 0] is float32
            # jt.unique requires consistent int32 input to avoid CUDA compilation errors
            # Use cast() instead of int() for explicit type conversion that works across precision modes
            i = i.cast(jt.int32)

            # jt.unique with return_counts=True alone returns Var, need return_inverse=True to get tuple
            _, _, counts = jt.unique(i, return_inverse=True, return_counts=True)
            counts = counts.to(dtype=jt.int32)
            max_count = counts.max().item()

            batch_data = []
            for j in range(batch_size):
                matches = i == j
                n = matches.sum().item()
                if n > 0:
                    indices = matches.where()[0]
                    selected = targets[indices, 1:]  # Direct indexing and slicing
                    
                    if selected.ndim == 1:
                        selected = selected.unsqueeze(0)
                    
                    actual_n = selected.shape[0]
                    
                    # Pad or truncate to max_count
                    if actual_n < max_count:
                        padding = jt.zeros((max_count - actual_n, ne - 1), dtype=jt.float32)
                        selected = jt.concat([selected, padding], dim=0)
                    elif actual_n > max_count:
                        selected = selected[:max_count]
                else:
                    selected = jt.zeros((max_count, ne - 1), dtype=jt.float32)
                
                # Ensure final shape is (max_count, ne-1)
                assert selected.shape[0] == max_count, f"Shape mismatch: {selected.shape[0]} vs {max_count}"
                batch_data.append(selected.unsqueeze(0))
            
            out = jt.concat(batch_data, dim=0)
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj)
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        
        # Verify anchor_points and pred_dist have matching anchor counts
        if pred_dist.ndim == 3 and anchor_points.ndim == 2:
            expected_anchors = pred_dist.shape[1]
            actual_anchors = anchor_points.shape[0]
            if actual_anchors != expected_anchors:
                # Log warning but let dist2bbox handle the shape mismatch
                from nkyolo.utils import LOGGER
                LOGGER.warning(
                    f"Anchor count mismatch in bbox_decode: anchor_points has {actual_anchors} anchors, "
                    f"but pred_dist has {expected_anchors} anchors. This may indicate an issue with "
                    f"make_anchors or pred_distri generation."
                )
        
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = jt.zeros(3)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        
        # self.stride is already jt.Var (ensured in __init__)
        # Ensure stride matches feats length
        stride_to_use = self.stride
        stride_len = len(stride_to_use)
        
        # If stride has fewer elements than feats, compute missing strides from feature shapes
        if stride_len < len(feats):
            LOGGER.warning(
                f"Stride length ({stride_len}) < feats length ({len(feats)}). "
                f"Computing missing strides from feature shapes."
            )
            # Compute stride from feature map sizes
            # Stride is typically: input_size / feature_map_size
            # We'll use a reference input size (640) and compute stride for each feature map
            reference_size = 640.0  # Standard YOLO input size
            additional_strides = []
            for i in range(stride_len, len(feats)):
                h = feats[i].shape[2]  # feature map height
                # Stride = input_size / feature_map_size
                computed_stride = reference_size / h
                additional_strides.append(computed_stride)
            # Concatenate existing strides with computed ones
            stride_to_use = jt.concat([stride_to_use, jt.Var(additional_strides)])
        
        # If stride has more elements than feats, truncate
        if stride_len > len(feats):
            LOGGER.warning(
                f"Stride length ({stride_len}) > feats length ({len(feats)}). "
                f"Truncating stride to match feats length."
            )
            stride_to_use = stride_to_use[:len(feats)]
        
        # Final verification
        if len(stride_to_use) != len(feats):
            LOGGER.error(
                f"Unable to resolve stride length mismatch: feats has {len(feats)} elements, "
                f"but stride has {len(stride_to_use)} elements after adjustment."
            )
            raise ValueError(
                f"feats length ({len(feats)}) must match stride length ({len(stride_to_use)})"
            )
        
        # Generate pred_distri and pred_scores
        # Each feat is reshaped to (batch, no, h*w), then concatenated along dim=2
        # Result: (batch, no, total_h*w) where total_h*w is sum of all h*w
        pred_distri, pred_scores = jt.concat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        # Get stride value for image size calculation - use getitem() instead of float()
        stride_val = stride_to_use.getitem(0)
        imgsz = jt.array(list(feats[0].shape[2:]),dtype = dtype) * stride_val  # image size (h,w)
        
        # Generate anchor_points - should have same total count as pred_distri's second dimension
        # Pass jt.Var directly, no type conversion
        anchor_points, stride_tensor = make_anchors(feats, stride_to_use, 0.5)
        
        # Verify anchor count matches pred_distri
        expected_anchors = pred_distri.shape[1]  # (batch, anchors, channels)
        actual_anchors = anchor_points.shape[0]   # (anchors, 2)
        if actual_anchors != expected_anchors:
            from nkyolo.utils import LOGGER
            LOGGER.error(
                f"Anchor count mismatch: anchor_points has {actual_anchors} anchors, "
                f"but pred_distri has {expected_anchors} anchors. "
                f"This indicates a bug in make_anchors or pred_distri generation. "
                f"Feats shapes: {[f.shape for f in feats]}, "
                f"Strides: {self.stride}"
            )
            raise ValueError(
                f"Anchor count mismatch: anchor_points ({actual_anchors}) != pred_distri ({expected_anchors}). "
                f"This should not happen if make_anchors and pred_distri use the same feats."
            )

        # Targets
        # Critical: Ensure batch_idx is int32 before concat to avoid type promotion issues
        # When model switches from float16 (epoch 1) to float32 (epoch 2+ after validation),
        # bboxes dtype changes, which would promote batch_idx to float if not explicitly set to int32
        batch_idx = jt.array(batch["batch_idx"], dtype=jt.int32).view(-1, 1)
        cls = jt.array(batch["cls"], dtype=jt.int32).view(-1, 1)
        bboxes = jt.array(batch["bboxes"]).view(-1, 4)
        # Concat will promote int32 to float, but we'll extract batch_idx separately in preprocess
        targets = jt.concat((
            batch_idx.cast(bboxes.dtype),  # Cast to match bboxes dtype for concat
            cls.cast(bboxes.dtype),        # Cast to match bboxes dtype for concat
            bboxes,
        ), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        # Rename gt_ to gt
        mask_gt = gt_bboxes.sum(2, keepdim=True) > 0.0

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).cast(dtype),   
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = jt.maximum(target_scores.sum(), 1.0)
        # print("target_scores_sum:", target_scores_sum)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        # loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        # loss_cls_allqwq = fix_manual_bce_with_logits(pred_scores, target_scores.to(dtype))
        loss[1] = fix_manual_bce_with_logits(pred_scores, target_scores.cast(dtype)).sum() / target_scores_sum

        # Bbox loss
        if (fg_mask > 0).sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            # Keep bbox branch in graph so all ranks produce identical grad/allreduce sets in DDP.
            loss[0] += (pred_distri * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # total loss, loss items
        # return loss.sum(), loss.detach()


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        # self.overlap = model.args.overlap_mask
        self.overlap = bool(getattr(model.args, "overlap_mask", False))

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = jt.zeros(4)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = jt.concat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        # imgsz = jt.Var(feats[0].shape[2:], dtype=dtype) * self.stride[0]  # image size (h,w)
        imgsz = jt.array(list(feats[0].shape[2:]), dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = jt.concat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True) > 0.0

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).cast(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # target_scores_sum = max(target_scores.sum(), 1)
        target_scores_sum = jt.maximum(target_scores.sum(), 1.0)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        # loss[2] = self.bce(pred_scores, target_scores.cast(dtype)).sum() / target_scores_sum  # BCE
        _loss_bce = jt.maximum(pred_scores, 0.0) - pred_scores * target_scores.cast(dtype) + jt.log(1.0 + jt.exp(-jt.abs(pred_scores)))
        loss[2] = _loss_bce.sum() / target_scores_sum

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = nn.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            # Also touch pred_distri to keep bbox branch grads present on all ranks.
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum() + (pred_distri * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # total loss, loss items

    @staticmethod
    def single_mask_loss(
        gt_mask: jt.Var, pred: jt.Var, proto: jt.Var, xyxy: jt.Var, area: jt.Var
    ) -> jt.Var:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (jt.Var): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (jt.Var): Predicted mask coefficients of shape (n, 32).
            proto (jt.Var): Prototype masks of shape (32, H, W).
            xyxy (jt.Var): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (jt.Var): Area of each ground truth bounding box of shape (n,).

        Returns:
            (jt.Var): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        # pred_mask = jt.matmul(pred.unsqueeze(1), proto.unsqueeze(0))  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)

        # loss = jt.nn.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")

        n, c = pred.shape
        h, w = proto.shape[-2:]
        
        
        pred_mask = jt.matmul(pred, proto.view(c, -1)).view(n, h, w)

        loss = jt.maximum(pred_mask, 0.0) - pred_mask * gt_mask + jt.log(1.0 + jt.exp(-jt.abs(pred_mask)))


        return (crop_mask(loss, xyxy).mean(dims=(1, 2)) / area).sum()
        
        # return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: jt.Var,
        masks: jt.Var,
        target_gt_idx: jt.Var,
        target_bboxes: jt.Var,
        batch_idx: jt.Var,
        proto: jt.Var,
        pred_masks: jt.Var,
        imgsz: jt.Var,
        overlap: bool,
    ) -> jt.Var:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (jt.Var): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (jt.Var): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (jt.Var): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (jt.Var): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (jt.Var): Batch indices of shape (N_labels_in_batch, 1).
            proto (jt.Var): Prototype masks of shape (BS, 32, H, W).
            pred_masks (jt.Var): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (jt.Var): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (jt.Var): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        # mxyxy = target_bboxes_normalized * jt.Var([mask_w, mask_h, mask_w, mask_h])
        mxyxy = target_bboxes_normalized * jt.array([mask_w, mask_h, mask_w, mask_h])

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = jt.array(OKS_SIGMA, dtype=jt.float32) if is_pose else jt.ones(nkpt, dtype=jt.float32) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = jt.zeros(5)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = jt.concat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        # imgsz = jt.Var(feats[0].shape[2:], dtype=dtype) * self.stride[0]  # image size (h,w)
        imgsz = jt.array(list(feats[0].shape[2:]), dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = jt.concat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True) > 0.0


        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)
        pred_kpts = pred_kpts.reshape(batch_size, pred_kpts.shape[1], self.kpt_shape[0], self.kpt_shape[1])
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).cast(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # target_scores_sum = max(target_scores.sum(), 1)
        target_scores_sum = jt.maximum(target_scores.sum(), 1.0)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        # loss[3] = self.bce(pred_scores, target_scores.cast(dtype)).sum() / target_scores_sum  # BCE
        _loss_bce = jt.maximum(pred_scores, 0.0) - pred_scores * target_scores.cast(dtype) + jt.log(1.0 + jt.exp(-jt.abs(pred_scores)))
        loss[3] = _loss_bce.sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )
        else:
            # Keep bbox/keypoint branches in graph for DDP grad/allreduce parity.
            loss[0] += (pred_distri * 0).sum()
            loss[1] += (pred_kpts * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # total loss, loss items

    # @staticmethod
    def kpts_decode(self, anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (jt.Var): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (jt.Var): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (jt.Var): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (jt.Var): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (jt.Var): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (jt.Var): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (jt.Var): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (jt.Var): The keypoints loss.
                - kpts_obj_loss (jt.Var): The keypoints object loss.
        """
        batch_idx: jt.Var = batch_idx.flatten()
        # Ensure int32 dtype to avoid compilation conflicts in float16 mode
        # Use explicit cast to int32 to avoid CUDA compilation type inference issues
        # This is critical when batch_idx may have been promoted to float16/float32
        if batch_idx.dtype != jt.int32:
            batch_idx = batch_idx.cast(jt.int32)
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        # jt.unique with return_counts=True alone returns Var, need return_inverse=True to get tuple
        _, _, counts = jt.unique(batch_idx, return_inverse=True, return_counts=True)
        max_kpts = counts.max().item()

        # Create a tensor to hold batched keypoints
        batched_keypoints = jt.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2])
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else jt.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = nn.cross_entropy_loss(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = jt.zeros(batch_size, 0, 6)
        else:
            i = targets[:, 0]  # image index
            # Ensure int32 dtype to avoid compilation conflicts in float16 mode
            # Use explicit int() cast to avoid CUDA compilation type inference issues
            if i.dtype != jt.int32:
                i = i.int()
            # jt.unique with return_counts=True alone returns Var, need return_inverse=True to get tuple
            _, _, counts = jt.unique(i, return_inverse=True, return_counts=True)
            counts = counts.to(dtype=jt.int32)
            max_count = counts.max().item()
            
            batch_data = []
            for j in range(batch_size):
                matches = i == j
                n = matches.sum().item()
                if n > 0:
                    indices = matches.where()[0]
                    selected = targets[indices]
                    
                    if len(selected.shape) == 1:
                        selected = selected.unsqueeze(0)
                    
                    bboxes = selected[:, 2:]  
                    bboxes[..., :4].mul_(scale_tensor)
                    combined = jt.concat([selected[:, 1:2], bboxes], dim=-1)
                    
                    actual_n = combined.shape[0]
                    
                    if actual_n < max_count:
                        padding = jt.zeros((max_count - actual_n, 6), dtype=jt.float32)
                        combined = jt.concat([combined, padding], dim=0)
                    elif actual_n > max_count:
                        combined = combined[:max_count]
                else:
                    combined = jt.zeros((max_count, 6), dtype=jt.float32)
                
                assert combined.shape[0] == max_count, f"Shape mismatch: {combined.shape[0]} vs {max_count}"
                batch_data.append(combined.unsqueeze(0))
            
            out = jt.concat(batch_data, dim=0)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = jt.zeros(3)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = jt.concat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        # imgsz = jt.Var(feats[0].shape[2:] , dtype=dtype) * self.stride[0]  # image size (h,w)
        imgsz = jt.array(list(feats[0].shape[2:]), dtype=dtype) * self.stride[0]

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = jt.concat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
        rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
        targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
        mask_gt = gt_bboxes.sum(2, keepdim=True) > 0.0

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.cast(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # target_scores_sum = max(target_scores.sum(), 1)
        target_scores_sum = jt.maximum(target_scores.sum(), 1.0)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        # loss[1] = self.bce(pred_scores, target_scores.cast(dtype)).sum() / target_scores_sum  # BCE
        _loss_bce = jt.maximum(pred_scores, 0.0) - pred_scores * target_scores.cast(dtype) + jt.log(1.0 + jt.exp(-jt.abs(pred_scores)))
        loss[1] = _loss_bce.sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            # Keep all OBB prediction branches in graph for DDP grad/allreduce parity.
            loss[0] += (pred_angle * 0).sum() + (pred_distri * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # total loss, loss items

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (jt.Var): Anchor points, (h*w, 2).
            pred_dist (jt.Var): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (jt.Var): Predicted angle, (bs, h*w, 1).

        Returns:
            (jt.Var): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.cast(pred_dist.dtype))
        return jt.concat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, **kwargs):
        params = model.parameters()
        if params:
            device = "cuda" if jt.flags.use_cuda else "cpu"
        else:
            device = "cpu"
        self.device = device
        self.nc = model.nc  # number of classes
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
