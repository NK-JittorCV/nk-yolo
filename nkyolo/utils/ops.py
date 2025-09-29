# NK-YOLO 🚀 AGPL-3.0 License
# Refer to https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py

import contextlib
import math
import re
import time

import cv2
import numpy as np
import jittor as jt
import jittor.nn as nn

from nkyolo.utils import LOGGER
from nkyolo.utils.metrics import batch_probiou


class Profile(contextlib.ContextDecorator):
    """
    YOLOv8 Profile class. Use as a decorator with @Profile() or as a context manager with 'with Profile():'.

    Example:
        ```python
        from nkyolo.utils.ops import Profile

        with Profile(device=device) as dt:
            pass  # slow operation here

        print(dt)  # prints "Elapsed time is 9.5367431640625e-07 s"
        ```
    """

    def __init__(self, t=0.0, device=None):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
            device (jt.device): Devices used for model inference. Defaults to None (cpu).
        """
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """Start timing."""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):  # noqa
        """Stop timing."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def __str__(self):
        """Returns a human-readable string representing the accumulated elapsed time in the profiler."""
        return f"Elapsed time is {self.t} s"

    def time(self):
        """Get current time."""
        if self.cuda:
            jt.cuda.synchronize(self.device)
        return time.time()


def segment2box(segment, width=640, height=640):
    """
    Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy).

    Args:
        segment (jt.Var): the segment label
        width (int): the width of the image. Defaults to 640
        height (int): The height of the image. Defaults to 640

    Returns:
        (np.ndarray): the minimum and maximum x and y values of the segment.
    """
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x = x[inside]
    y = y[inside]
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )  # xyxy


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
    specified in (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (jt.Var): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        xywh (bool): The box format is xywh or not, default=False.

    Returns:
        boxes (jt.Var): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def make_divisible(x, divisor):
    """
    Returns the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | jt.Var): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, jt.Var):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def compute_iou(box1, box2):
    """
    简单的 IoU 计算实现
    
    Args:
        box1 (jt.Var): 边界框1，shape (1, 4)，格式 xyxy
        box2 (jt.Var): 边界框2，shape (N, 4)，格式 xyxy
    
    Returns:
        jt.Var: IoU 值，shape (1, N)
    """
    # 逐个计算IoU，避免广播问题
    ious = []
    
    # 确保正确的形状和数据访问
    if box1.ndim == 2:
        box1_flat = box1.view(-1)  # flatten to 1D
    else:
        box1_flat = box1
    
    for i in range(box2.shape[0]):
        box2_i = box2[i]  # shape (4,)
        
        # 安全的数据访问方式
        try:
            # 尝试使用 numpy 转换
            box1_np = box1_flat.numpy()
            box2_np = box2_i.numpy()
            
            x1 = max(float(box1_np[0]), float(box2_np[0]))
            y1 = max(float(box1_np[1]), float(box2_np[1]))
            x2 = min(float(box1_np[2]), float(box2_np[2]))
            y2 = min(float(box1_np[3]), float(box2_np[3]))
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            
            # 计算面积
            area1 = (float(box1_np[2]) - float(box1_np[0])) * (float(box1_np[3]) - float(box1_np[1]))
            area2 = (float(box2_np[2]) - float(box2_np[0])) * (float(box2_np[3]) - float(box2_np[1]))
            
            # 计算IoU
            union = area1 + area2 - intersection
            iou = intersection / (union + 1e-6) if union > 0 else 0
            ious.append(iou)
            
        except:
            # 如果转换失败，使用默认值
            ious.append(0.0)
    
    return jt.array(ious).unsqueeze(0)  # shape (1, N)


def simple_nms(boxes, scores, iou_threshold):
    """
    简单的 NMS 实现，用于 Jittor 兼容性
    
    Args:
        boxes (jt.Var): 边界框，shape (N, 4)，格式 xyxy
        scores (jt.Var): 置信度分数，shape (N,)
        iou_threshold (float): IoU 阈值
    
    Returns:
        jt.Var: 保留的框的索引
    """
    if boxes.numel() == 0:
        return jt.empty((0,), dtype=jt.int64)
    
    # 按分数降序排序
    _, indices = scores.sort(descending=True)
    
    keep = []
    while len(indices) > 0:
        # 保留当前最高分数的框
        current = indices[0]
        keep.append(current.item())
        
        if len(indices) == 1:
            break
            
        # 计算当前框与其余框的 IoU
        current_box = boxes[current].unsqueeze(0)
        other_boxes = boxes[indices[1:]]
        
        # 计算 IoU - 使用自定义实现
        ious = compute_iou(current_box, other_boxes).squeeze(0)
        
        # 保留 IoU 小于阈值的框
        mask = ious <= iou_threshold
        indices = indices[1:][mask]
    
    return jt.array(keep, dtype=jt.int64)


def nms_rotated(boxes, scores, threshold=0.45):
    """
    NMS for oriented bounding boxes using probiou and fast-nms.

    Args:
        boxes (jt.Var): Rotated bounding boxes, shape (N, 5), format xywhr.
        scores (jt.Var): Confidence scores, shape (N,).
        threshold (float, optional): IoU threshold. Defaults to 0.45.

    Returns:
        (jt.Var): Indices of boxes to keep after NMS.
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)
    sorted_idx = jt.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes).triu_(diagonal=1)
    pick = jt.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
    return sorted_idx[pick]


# import jittor as jt

def jtnms(boxes: jt.Var, scores: jt.Var, iou_threshold: float) -> jt.Var:
    """
    Jittor 实现的非极大值抑制 (NMS)，用于去除重叠度高的检测框
    
    Args:
        boxes (jt.Var): [N, 4], 边界框坐标，格式为 (x1, y1, x2, y2)
        scores (jt.Var): [N] 或 [N, 1]，每个边界框的置信度分数
        iou_threshold (float): IOU 阈值，超过此阈值的框将被抑制
        
    Returns:
        jt.Var: 保留的框索引 [M,], dtype=int32
    """
    # 处理空输入
    if boxes.numel() == 0:
        return jt.array([], dtype='int32')
    
    # 确保 scores 是 [N] 形状
    if scores.ndim == 2:
        scores = scores.squeeze(1)  # [N, 1] -> [N]
    
    # 按 scores 降序排序并获取排序索引
    _, order = jt.argsort(scores, descending=True)
    boxes = boxes[order]  # 排序后的 boxes
    keep = jt.zeros(boxes.shape[0], dtype='bool')  # 用于标记保留的框
    num_keep = 0  # 已保留的框数量
    
    # 预先计算所有框的面积，避免重复计算
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    for i in range(boxes.shape[0]):
        # 如果当前框已被标记为抑制，则跳过
        if keep[i]:
            continue
            
        # 保留当前框
        keep[i] = True
        num_keep += 1
        
        # 计算当前框与剩余所有框的IOU
        # 当前框坐标
        x1, y1, x2, y2 = boxes[i]
        
        # 剩余框与当前框的交集坐标
        xx1 = jt.maximum(x1, boxes[i+1:, 0])
        yy1 = jt.maximum(y1, boxes[i+1:, 1])
        xx2 = jt.minimum(x2, boxes[i+1:, 2])
        yy2 = jt.minimum(y2, boxes[i+1:, 3])
        
        # 计算交集面积
        w = jt.maximum(0.0, xx2 - xx1)
        h = jt.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        # 计算并集面积和IOU
        union = areas[i] + areas[i+1:] - inter
        iou = inter / jt.maximum(union, 1e-9)  # 防止除零
        
        # 抑制IOU超过阈值的框
        overlap_mask = iou > iou_threshold
        keep[i+1:][overlap_mask] = False
    
    # 获取保留的索引并映射回原始顺序
    keep_indices = jt.where(keep)[0]
    return order[keep_indices].astype('int32')


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (jt.Var): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, jt.Var]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.

    Returns:
        (List[jt.Var]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = jt.Var(classes, device=prediction.device)

    if prediction.shape[-1] == 6:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].max(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = jt.concat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [jt.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # 修复 Jittor 布尔索引兼容性
        mask = xc[xi]
        if hasattr(mask, 'where'):
            indices = mask.where()[0]  # Jittor 方式
            x = x[indices]
        else:
            x = x[mask]  # 原始方式

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = jt.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = jt.concat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = jt.where(cls > conf_thres)
            x = jt.concat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)

        else:  # 仅保留最佳类别
            conf = cls.max(1, keepdim=True)
            argmax_result = jt.argmax(cls, dim=1)
            if isinstance(argmax_result, tuple):
                j = argmax_result[0]
            else:
                j = argmax_result
            j = j.unsqueeze(1) 
            filt = conf.view(-1) > conf_thres
            x = jt.cat((box, conf, j.float(), mask), 1)[filt]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = jt.concat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            # 1. 首先将scores合并到boxes中
            boxes_with_scores = jt.cat([boxes, scores.unsqueeze(1)], dim=1) 
            # 2. 调用Jittor的nms函数，传入boxes和iou阈值
            i = jt.nms(boxes_with_scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        # # Experimental
        # merge = False  # use merge-NMS
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     from .metrics import box_iou
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # IoU matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = jt.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     redundant = True  # require redundant detections
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
        boxes (jt.Var): the bounding boxes to clip
        shape (tuple): the shape of the image

    Returns:
        (jt.Var | numpy.ndarray): Clipped boxes
    """
    if isinstance(boxes, jt.Var):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.

    Args:
        coords (jt.Var | numpy.ndarray): A list of line coordinates.
        shape (tuple): A tuple of integers representing the size of the image in the format (height, width).

    Returns:
        (jt.Var | numpy.ndarray): Clipped coordinates
    """
    if isinstance(coords, jt.Var):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        coords[..., 0] = coords[..., 0].clamp(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clamp(0, shape[0])  # y
    else:  # np.array (faster grouped)
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    return coords


def scale_image(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size.

    Args:
        masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
        im0_shape (tuple): the original image shape
        ratio_pad (tuple): the ratio of the padding to the original image.

    Returns:
        masks (np.ndarray): The masks that are being returned with shape [h, w, num].
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        # gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | jt.Var): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | jt.Var): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    if isinstance(x, jt.Var):
        y = jt.zeros(x.shape, dtype=x.dtype)  # 显式指定设备和类型
    else:
        y = np.zeros_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | jt.Var): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | jt.Var): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = jt.empty(x.shape, dtype=x.dtype) if isinstance(x, jt.Var) else np.empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    Convert normalized bounding box coordinates to pixel coordinates.

    Args:
        x (np.ndarray | jt.Var): The bounding box coordinates.
        w (int): Width of the image. Defaults to 640
        h (int): Height of the image. Defaults to 640
        padw (int): Padding width. Defaults to 0
        padh (int): Padding height. Defaults to 0
    Returns:
        y (np.ndarray | jt.Var): The coordinates of the bounding box in the format [x1, y1, x2, y2] where
            x1,y1 is the top-left corner, x2,y2 is the bottom-right corner of the bounding box.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = jt.empty_like(x) if isinstance(x, jt.Var) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height, normalized) format. x, y,
    width and height are normalized to image dimensions.

    Args:
        x (np.ndarray | jt.Var): The input bounding box coordinates in (x1, y1, x2, y2) format.
        w (int): The width of the image. Defaults to 640
        h (int): The height of the image. Defaults to 640
        clip (bool): If True, the boxes will be clipped to the image boundaries. Defaults to False
        eps (float): The minimum value of the box's width and height. Defaults to 0.0

    Returns:
        y (np.ndarray | jt.Var): The bounding box coordinates in (x, y, width, height, normalized) format
    """
    if clip:
        x = clip_boxes(x, (h - eps, w - eps))
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = jt.empty_like(x) if isinstance(x, jt.Var) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def xywh2ltwh(x):
    """
    Convert the bounding box format from [x, y, w, h] to [x1, y1, w, h], where x1, y1 are the top-left coordinates.

    Args:
        x (np.ndarray | jt.Var): The input tensor with the bounding box coordinates in the xywh format

    Returns:
        y (np.ndarray | jt.Var): The bounding box coordinates in the xyltwh format
    """
    y = x.clone() if isinstance(x, jt.Var) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    return y


def xyxy2ltwh(x):
    """
    Convert nx4 bounding boxes from [x1, y1, x2, y2] to [x1, y1, w, h], where xy1=top-left, xy2=bottom-right.

    Args:
        x (np.ndarray | jt.Var): The input tensor with the bounding boxes coordinates in the xyxy format

    Returns:
        y (np.ndarray | jt.Var): The bounding box coordinates in the xyltwh format.
    """
    y = x.clone() if isinstance(x, jt.Var) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def ltwh2xywh(x):
    """
    Convert nx4 boxes from [x1, y1, w, h] to [x, y, w, h] where xy1=top-left, xy=center.

    Args:
        x (jt.Var): the input tensor

    Returns:
        y (np.ndarray | jt.Var): The bounding box coordinates in the xywh format.
    """
    y = x.clone() if isinstance(x, jt.Var) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # center x
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # center y
    return y


def xyxyxyxy2xywhr(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation]. Rotation values are
    returned in radians from 0 to pi/2.

    Args:
        x (numpy.ndarray | jt.Var): Input box corners [xy1, xy2, xy3, xy4] of shape (n, 8).

    Returns:
        (numpy.ndarray | jt.Var): Converted data in [cx, cy, w, h, rotation] format of shape (n, 5).
    """
    is_torch = isinstance(x, jt.Var)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
        # especially some objects are cut off by augmentations in dataloader.
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
    return jt.Var(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)


def xywhr2xyxyxyxy(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in radians from 0 to pi/2.

    Args:
        x (numpy.ndarray | jt.Var): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray | jt.Var): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    cos, sin, cat, stack = (
        (jt.cos, jt.sin, jt.concat, jt.stack)
        if isinstance(x, jt.Var)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def ltwh2xyxy(x):
    """
    It converts the bounding box from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.

    Args:
        x (np.ndarray | jt.Var): the input image

    Returns:
        y (np.ndarray | jt.Var): the xyxy coordinates of the bounding boxes.
    """
    y = x.clone() if isinstance(x, jt.Var) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]  # width
    y[..., 3] = x[..., 3] + x[..., 1]  # height
    return y


def segments2boxes(segments):
    """
    It converts segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh).

    Args:
        segments (list): list of segments, each segment is a list of points, each point is a list of x, y coordinates

    Returns:
        (np.ndarray): the xywh coordinates of the bounding boxes.
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    """
    Inputs a list of segments (n,2) and returns a list of segments (n,2) up-sampled to n points each.

    Args:
        segments (list): a list of (n,2) arrays, where n is the number of points in the segment.
        n (int): number of points to resample the segment to. Defaults to 1000

    Returns:
        segments (list): the resampled segments.
    """
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # segment xy
    return segments


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

    Args:
        masks (jt.Var): [n, h, w] tensor of masks
        boxes (jt.Var): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
        (jt.Var): The masks are being cropped to the bounding box.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = jt.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = jt.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = jt.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (jt.Var): A tensor of shape [mask_dim, mask_h, mask_w].
        masks_in (jt.Var): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (jt.Var): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.

    Returns:
        (jt.Var): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # CHW
    width_ratio = mw / iw
    height_ratio = mh / ih

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio
    downsampled_bboxes[:, 1] *= height_ratio

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = nn.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    return masks.gt_(0.0)


def process_mask_native(protos, masks_in, bboxes, shape):
    """
    It takes the output of the mask head, and crops it after upsampling to the bounding boxes.

    Args:
        protos (jt.Var): [mask_dim, mask_h, mask_w]
        masks_in (jt.Var): [n, mask_dim], n is number of masks after nms
        bboxes (jt.Var): [n, 4], n is number of masks after nms
        shape (tuple): the size of the input image (h,w)

    Returns:
        masks (jt.Var): The returned masks with dimensions [h, w, n]
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = scale_masks(masks[None], shape)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.0)


def scale_masks(masks, shape, padding=True):
    """
    Rescale segment masks to shape.

    Args:
        masks (jt.Var): (N, C, H, W).
        shape (tuple): Height and width.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
    """
    mh, mw = masks.shape[2:]
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]  # wh padding
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)  # y, x
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
    masks = masks[..., top:bottom, left:right]

    masks = nn.interpolate(masks, shape, mode="bilinear", align_corners=False)  # NCHW
    return masks


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    Rescale segment coordinates (xy) from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): The shape of the image that the coords are from.
        coords (jt.Var): the coords to be scaled of shape n,2.
        img0_shape (tuple): the shape of the image that the segmentation is being applied to.
        ratio_pad (tuple): the ratio of the image size to the padded image size.
        normalize (bool): If True, the coordinates will be normalized to the range [0, 1]. Defaults to False.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        coords (jt.Var): The scaled coordinates.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords


def regularize_rboxes(rboxes):
    """
    Regularize rotated boxes in range [0, pi/2].

    Args:
        rboxes (jt.Var): Input boxes of shape(N, 5) in xywhr format.

    Returns:
        (jt.Var): The regularized boxes.
    """
    x, y, w, h, t = rboxes.unbind(dim=-1)
    # Swap edge and angle if h >= w
    w_ = jt.where(w > h, w, h)
    h_ = jt.where(w > h, h, w)
    t = jt.where(w > h, t, t + math.pi / 2) % math.pi
    return jt.stack([x, y, w_, h_, t], dim=-1)  # regularized boxes


def masks2segments(masks, strategy="largest"):
    """
    It takes a list of masks(n,h,w) and returns a list of segments(n,xy).

    Args:
        masks (jt.Var): the output of the model, which is a tensor of shape (batch_size, 160, 160)
        strategy (str): 'concat' or 'largest'. Defaults to largest

    Returns:
        segments (List): list of segment masks
    """
    segments = []
    for x in masks.int().cpu().numpy().astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "concat":  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == "largest":  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments


def convert_torch2numpy_batch(batch: jt.Var) -> np.ndarray:
    """
    Convert a batch of FP32 jt tensors (0.0-1.0) to a NumPy uint8 array (0-255), changing from BCHW to BHWC layout.

    Args:
        batch (jt.Var): Input tensor batch of shape (Batch, Channels, Height, Width) and dtype jt.float32.

    Returns:
        (np.ndarray): Output NumPy array batch of shape (Batch, Height, Width, Channels) and dtype uint8.
    """
    return (batch.permute(0, 2, 3, 1).contiguous() * 255).clamp(0, 255).to(jt.uint8).cpu().numpy()


def clean_str(s):
    """
    Cleans a string by replacing special characters with '_' character.

    Args:
        s (str): a string needing special characters replaced

    Returns:
        (str): a string with special characters replaced by an underscore _
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def compute_iou_optimized(box1, box2):
    """
    计算两个单独边界框的IoU，专门优化处理Jittor Var对象。
    
    Args:
        box1 (jt.Var): 边界框1，格式为[x1, y1, x2, y2]
        box2 (jt.Var): 边界框2，格式为[x1, y1, x2, y2]
    
    Returns:
        float: IoU值
    """
    # 转换为numpy以避免Jittor broadcasting问题
    if hasattr(box1, 'numpy'):
        box1_np = box1.numpy()
    else:
        box1_np = np.array(box1)
    
    if hasattr(box2, 'numpy'):
        box2_np = box2.numpy()
    else:
        box2_np = np.array(box2)
    
    # 确保是1D数组
    box1_np = box1_np.flatten()
    box2_np = box2_np.flatten()
    
    # 计算交集区域
    x1 = max(box1_np[0], box2_np[0])
    y1 = max(box1_np[1], box2_np[1])
    x2 = min(box1_np[2], box2_np[2])
    y2 = min(box1_np[3], box2_np[3])
    
    # 检查是否有交集
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # 计算交集面积
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算各自面积
    area1 = (box1_np[2] - box1_np[0]) * (box1_np[3] - box1_np[1])
    area2 = (box2_np[2] - box2_np[0]) * (box2_np[3] - box2_np[1])
    
    # 计算并集面积
    union = area1 + area2 - intersection
    
    # 避免除零
    if union <= 0:
        return 0.0
    
    return intersection / union


def simple_nms(boxes, scores, iou_threshold):
    """
    简单的NMS实现，作为jt.ops.nms的fallback。
    
    Args:
        boxes (jt.Var): 边界框，形状为[N, 4]，格式为[x1, y1, x2, y2]
        scores (jt.Var): 置信度分数，形状为[N]
        iou_threshold (float): IoU阈值
    
    Returns:
        jt.Var: 保留的框的索引
    """
    # Input validation
    if not (hasattr(boxes, "ndim") and hasattr(boxes, "shape")):
        raise ValueError("boxes must be a Jittor Var or array-like with .ndim and .shape attributes")
    if not (hasattr(scores, "ndim") and hasattr(scores, "shape")):
        raise ValueError("scores must be a Jittor Var or array-like with .ndim and .shape attributes")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must be a 2D tensor with shape [N, 4], but got shape {boxes.shape}")
    if scores.ndim != 1:
        raise ValueError(f"scores must be a 1D tensor with shape [N], but got shape {scores.shape}")
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(f"boxes and scores must have the same number of elements in the first dimension, but got {boxes.shape[0]} and {scores.shape[0]}")
    if boxes.shape[0] == 0:
        return jt.array([], dtype=jt.int64)
    
    # 按分数降序排序
    if hasattr(scores, 'argsort'):
        argsort_result = scores.argsort(descending=True)
        # Jittor的argsort返回(indices, sorted_values)元组
        if isinstance(argsort_result, tuple):
            sorted_indices = argsort_result[0]  # 只取索引
        else:
            sorted_indices = argsort_result
    else:
        # fallback for older jittor versions
        argsort_result = jt.argsort(scores, descending=True)
        if isinstance(argsort_result, tuple):
            sorted_indices = argsort_result[0]
        else:
            sorted_indices = argsort_result
    
    keep = []
    
    while sorted_indices.shape[0] > 0:
        # 选择分数最高的框
        current = sorted_indices[0]
        # 安全地获取索引值
        if hasattr(current, 'numpy'):
            current_np = current.numpy()
            if current_np.size == 1:
                current_idx = int(current_np.item())
            else:
                current_idx = int(current_np[0])
        else:
            current_idx = int(current)
        keep.append(current_idx)
        
        if sorted_indices.shape[0] == 1:
            break
        
        # 计算当前框与剩余框的IoU
        current_box = boxes[current_idx]
        remaining_indices = sorted_indices[1:]
        remaining_boxes = boxes[remaining_indices]
        
        # 计算IoU
        ious = []
        for i in range(remaining_boxes.shape[0]):
            iou = compute_iou_optimized(current_box, remaining_boxes[i])
            ious.append(iou)
        
        # 过滤掉IoU大于阈值的框
        ious = np.array(ious)
        mask = ious <= iou_threshold
        
        # 更新剩余索引
        if np.any(mask):
            remaining_indices_np = remaining_indices.numpy() if hasattr(remaining_indices, 'numpy') else remaining_indices
            kept_indices = remaining_indices_np[mask]
            sorted_indices = jt.array(kept_indices)
        else:
            break
    
    return jt.array(keep, dtype=jt.int64)