import os

# Jittor 在 import 时若发现 nvcc，会编译并链接 cuDNN；仅安装 CUDA Toolkit、未装 cuDNN 开发包时
# 会报错：cudnn.h not found / CUDA found but cudnn is not loaded。
# 解决：① 按 NVIDIA 文档安装与 CUDA 版本匹配的 cuDNN（含 cudnn.h）；
#      ② 或运行：NKYOLO_CPU=1 python train.py（仅 CPU，跳过 CUDA 初始化）。
_force_cpu = os.environ.get("NKYOLO_CPU", "0") == "1"
if _force_cpu:
    os.environ["nvcc_path"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["use_cuda"] = "0"
    os.environ["JT_USE_CUDA"] = "0"

from nkyolo import YOLO

import jittor as jt

if _force_cpu:
    jt.flags.use_cuda = 0

# batch=128 @640 易在 ~24GB 显存上 OOM（EMA/AMP/优化器占用大）。默认 32；CPU 用 8。
# 需要更大 batch 可：NKYOLO_BATCH=64 python train.py
_batch = int(os.environ.get("NKYOLO_BATCH", "8" if _force_cpu else "32"))

model = YOLO("yolov10n.yaml")
train_kw = dict(data="coco128.yaml", epochs=10, batch=_batch, amp=True, val_amp=True)
if _force_cpu:
    train_kw["device"] = "cpu"
model.train(**train_kw)
