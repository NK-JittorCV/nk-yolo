# jittoryolo YOLO 🚀, AGPL-3.0 license

import gc
import math
import os
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import jittor as jt
import jittor.nn as nn
import jittor.nn as F

from jittoryolo.utils import (
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_KEYS,
    LOGGER,
    NUM_THREADS,
    PYTHON_VERSION,
    TORCHVISION_VERSION,
    WINDOWS,
    __version__,
    colorstr,
)
from jittoryolo.utils.checks import check_version

try:
    import thop
except ImportError:
    thop = None

# Version checks (all default to version>=min_version)
TORCH_1_9 = check_version(jt.__version__, "1.9.0")
TORCH_1_13 = check_version(jt.__version__, "1.13.0")
TORCH_2_0 = check_version(jt.__version__, "2.0.0")
TORCH_2_4 = check_version(jt.__version__, "2.4.0")
TORCHVISION_0_10 = check_version(TORCHVISION_VERSION, "0.10.0")
TORCHVISION_0_11 = check_version(TORCHVISION_VERSION, "0.11.0")
TORCHVISION_0_13 = check_version(TORCHVISION_VERSION, "0.13.0")
TORCHVISION_0_18 = check_version(TORCHVISION_VERSION, "0.18.0")
if WINDOWS and check_version(jt.__version__, "==2.4.0"):  # reject version 2.4.0 on Windows
    LOGGER.warning(
        "WARNING ⚠️ Known issue with torch==2.4.0 on Windows with CPU, recommend upgrading to torch>=2.4.1 to resolve "
        "https://github.com/jittoryolo/jittoryolo/issues/15049"
    )


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Ensures all processes in distributed training wait for the local master (rank 0) to complete a task first."""
    initialized = dist.is_available() and dist.is_initialized()

    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[local_rank])
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[local_rank])


def smart_inference_mode():
    """Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator."""

    def decorate(fn):
        """Applies appropriate torch decorator for inference mode based on torch version."""
        if TORCH_1_9 and torch.is_inference_mode_enabled():
            return fn  # already in inference_mode, act as a pass-through
        else:
            return (torch.inference_mode if TORCH_1_9 else torch.no_grad)()(fn)

    return decorate


@contextmanager
def autocast(enabled: bool, device: str = "cuda"):
    prev_amp = jt.flags.use_cuda_amp if hasattr(jt.flags, "use_cuda_amp") else 0
    if enabled and device == "cuda":
        jt.flags.use_cuda_amp = 1
    try:
        yield
    finally:
        if hasattr(jt.flags, "use_cuda_amp"):
            jt.flags.use_cuda_amp = prev_amp


def get_cpu_info():
    """Return a string with system CPU information, i.e. 'Apple M2'."""
    from jittoryolo.utils import PERSISTENT_CACHE  # avoid circular import error

    if "cpu_info" not in PERSISTENT_CACHE:
        try:
            import cpuinfo  # pip install py-cpuinfo

            k = "brand_raw", "hardware_raw", "arch_string_raw"  # keys sorted by preference
            info = cpuinfo.get_cpu_info()  # info dict
            string = info.get(k[0] if k[0] in info else k[1] if k[1] in info else k[2], "unknown")
            PERSISTENT_CACHE["cpu_info"] = string.replace("(R)", "").replace("CPU ", "").replace("@ ", "")
        except Exception:
            pass
    return PERSISTENT_CACHE.get("cpu_info", "unknown")


def get_gpu_info(index):
    """Return a string with system GPU information, i.e. 'Tesla T4, 15102MiB'."""
    properties = torch.cuda.get_device_properties(index)
    return f"{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB"


def select_device(device="", batch=0, newline=False, verbose=True):
    """
    Select appropriate Jittor device for running.
    
    Args:
        device (str, optional): Device string. Available options: "", "cpu", "cuda", "0", etc.
           Default empty string will auto-select first available GPU, or CPU if no GPU exists.
        batch (int, optional): Batch size used in model. Defaults to 0.
        newline (bool, optional): If True, add newline at end of log string. Defaults to False.
        verbose (bool, optional): If True, print device info. Defaults to True.
    
    Returns:
        str: Selected device string ("cpu" or "cuda")
    
    Raises:
        ValueError: If requested device is not available, or batch size is not multiple of GPU count in multi-GPU mode.
    """
    s = f"jittoryolo {__version__} 🚀 Python-{PYTHON_VERSION} Jittor-{jt.__version__}"

    # Process device string
    device = str(device).lower().strip()
    # Remove unnecessary chars 
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")

    # CPU mode
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        if verbose:
            LOGGER.info(f"{s} CPU Mode {'↵' if newline else ''}")
        return "cpu"

    # GPU mode
    if device:  # specific device requested
        if device == "cuda":
            device = "0"
        if "," in device:  # multi-GPU case
            device = ",".join(x for x in device.split(",") if x)  # clean sequential commas
            
        # Set visible GPUs
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        
        # Check CUDA availability
        if not jt.has_cuda:
            LOGGER.info(s)
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested. "
                f"Use 'device=cpu' or pass valid CUDA device(s) if available, "
                f"i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\njt.has_cuda: {jt.has_cuda}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}"
            )
            
        # Check batch size for multi-GPU
        n = len(device.split(","))
        if n > 1 and batch > 0:  # multi-GPU training
            if batch < n:
                raise ValueError(
                    f"Batch size {batch} must be larger than GPU count {n} for Multi-GPU training"
                )
            if batch % n != 0:
                raise ValueError(
                    f"Batch size {batch} must be divisible by GPU count {n} for Multi-GPU training. "
                    f"Try using batch size {batch // n * n} or {batch // n * n + n}"
                )
    
    # Default to CUDA if available
    if jt.has_cuda:
        device = "cuda"
        if verbose:
            LOGGER.info(f"{s} CUDA enabled {'↵' if newline else ''}")
    else:  # fallback to CPU
        device = "cpu" 
        if verbose:
            LOGGER.info(f"{s} CPU Mode {'↵' if newline else ''}")

    jt.flags.use_cuda = (device == "cuda")
    return device


def time_sync():
    """PyTorch-accurate time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
    )

    # Prepare filters
    w_conv = conv.weight.view(conv.out_channels, -1)
    w_bn = jt.diag(bn.weight.div(jt.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight = jt.matmul(w_bn, w_conv).view(fusedconv.weight.shape)

    # Prepare spatial bias
    b_conv = jt.zeros(conv.weight.shape[0]) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(jt.sqrt(bn.running_var + bn.eps))
    fusedconv.bias = jt.matmul(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn

    return fusedconv


def fuse_deconv_and_bn(deconv, bn):
    """Fuse ConvTranspose2d() and BatchNorm2d() layers."""
    fuseddconv = (
        nn.ConvTranspose2d(
            deconv.in_channels,
            deconv.out_channels,
            kernel_size=deconv.kernel_size,
            stride=deconv.stride,
            padding=deconv.padding,
            output_padding=deconv.output_padding,
            dilation=deconv.dilation,
            groups=deconv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(deconv.weight.device)
    )

    # Prepare filters
    w_deconv = deconv.weight.view(deconv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fuseddconv.weight.copy_(torch.mm(w_bn, w_deconv).view(fuseddconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(deconv.weight.shape[1], device=deconv.weight.device) if deconv.bias is None else deconv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fuseddconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fuseddconv


def model_info(model, detailed=False, verbose=True, imgsz=640):
    """
    Model information.

    imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320].
    """
    if not verbose:
        return
    n_p = get_num_params(model)  # number of parameters
    n_g = get_num_gradients(model)  # number of gradients
    n_l = len(list(model.modules()))  # number of layers
    if detailed:
        LOGGER.info(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}"
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            LOGGER.info(
                "%5g %40s %9s %12g %20s %10.3g %10.3g %10s"
                % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std(), p.dtype)
            )

    flops = get_flops(model, imgsz)
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(model, "yaml_file", "") or getattr(model, "yaml", {}).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    LOGGER.info(f"{model_name} summary{fused}: {n_l:,} layers, {n_p:,} parameters, {n_g:,} gradients{fs}")
    return n_l, n_p, n_g, flops


def get_num_params(model):
    """Return the total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    """Return the total number of parameters with gradients in a YOLO model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def model_info_for_loggers(trainer):
    """
    Return model info dict with useful model information.

    Example:
        YOLOv8n info for loggers
        ```python
        results = {
            "model/parameters": 3151904,
            "model/GFLOPs": 8.746,
            "model/speed_ONNX(ms)": 41.244,
            "model/speed_TensorRT(ms)": 3.211,
            "model/speed_PyTorch(ms)": 18.755,
        }
        ```
    """
    if trainer.args.profile:  # profile ONNX and TensorRT times
        from jittoryolo.utils.benchmarks import ProfileModels

        results = ProfileModels([trainer.last], device=trainer.device).profile()[0]
        results.pop("model/name")
    else:  # only return PyTorch times from most recent validation
        results = {
            "model/parameters": get_num_params(trainer.model),
            "model/GFLOPs": round(get_flops(trainer.model), 3),
        }
    results["model/speed_PyTorch(ms)"] = round(trainer.validator.speed["inference"], 3)
    return results


def get_flops(model, imgsz=640):
    """Return a YOLO model's FLOPs."""
    if not thop:
        return 0.0  # if not installed return 0.0 GFLOPs

    try:
        model = de_parallel(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]  # expand if int/float
        try:
            # Use stride size for input tensor
            stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32  # max stride
            im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
            flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # stride GFLOPs
            return flops * imgsz[0] / stride * imgsz[1] / stride  # imgsz GFLOPs
        except Exception:
            # Use actual image size for input tensor (i.e. required for RTDETR models)
            im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # input image in BCHW format
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # imgsz GFLOPs
    except Exception:
        return 0.0


def get_flops_with_torch_profiler(model, imgsz=640):
    """Compute model FLOPs (thop package alternative, but 2-10x slower unfortunately)."""
    if not TORCH_2_0:  # torch profiler implemented in torch>=2.0
        return 0.0
    model = de_parallel(model)
    p = next(model.parameters())
    if not isinstance(imgsz, list):
        imgsz = [imgsz, imgsz]  # expand if int/float
    try:
        # Use stride size for input tensor
        stride = (max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32) * 2  # max stride
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
        with torch.profiler.profile(with_flops=True) as prof:
            model(im)
        flops = sum(x.flops for x in prof.key_averages()) / 1e9
        flops = flops * imgsz[0] / stride * imgsz[1] / stride  # 640x640 GFLOPs
    except Exception:
        # Use actual image size for input tensor (i.e. required for RTDETR models)
        im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # input image in BCHW format
        with torch.profiler.profile(with_flops=True) as prof:
            model(im)
        flops = sum(x.flops for x in prof.key_averages()) / 1e9
    return flops


def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """Scales and pads an image tensor, optionally maintaining aspect ratio and padding to gs multiple."""
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    """Copies attributes from object 'b' to object 'a', with options to include/exclude certain attributes."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def get_latest_opset():
    """Return the second-most recent ONNX opset version supported by this version of PyTorch, adjusted for maturity."""
    if TORCH_1_13:
        # If the PyTorch>=1.13, dynamically compute the latest opset minus one using 'symbolic_opset'
        return max(int(k[14:]) for k in vars(torch.onnx) if "symbolic_opset" in k) - 1
    # Otherwise for PyTorch<=1.12 return the corresponding predefined opset
    version = torch.onnx.producer_version.rsplit(".", 1)[0]  # i.e. '2.3'
    return {"1.12": 15, "1.11": 14, "1.10": 13, "1.9": 12, "1.8": 12}.get(version, 12)


def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def is_parallel(model):
    """
    Returns True if model is parallelized in Jittor.
    Note: Currently just return False since Jittor's parallelization is different from PyTorch.
    """
    return False  # TODO: check if model is parallelized in Jittor

def de_parallel(model):
    """
    De-parallelize a model. In Jittor this is a pass-through for now.
    
    Args:
        model: A Jittor model.
        
    Returns:
        The same model since Jittor handles parallelization differently.
    """
    return model  # Currently just return model as is for Jittor


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Returns a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf."""
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1


def init_seeds(seed=0, deterministic=False):
    """Initialize RNG seeds and configure deterministic settings for Jittor."""
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set Jittor's global seed
    jt.set_global_seed(seed)
    
    # Configure deterministic settings
    if deterministic:
        # Set CUBLAS workspace configuration for deterministic behavior
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # Set Python's hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        # Enable deterministic algorithms (Jittor-specific logic may vary)
        # Note: Jittor may not have a direct equivalent to torch.use_deterministic_algorithms
        # You can set additional environment variables or configurations if available
    else:
        # Reset deterministic settings (if applicable)
        if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
            del os.environ["CUBLAS_WORKSPACE_CONFIG"]
        if "PYTHONHASHSEED" in os.environ:
            del os.environ["PYTHONHASHSEED"]


class ModelEMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models.
    Keeps a moving average of everything in the model state_dict (parameters and buffers).
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Initialize EMA for 'model' with given arguments."""
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp
        
        # Disable gradient computation for EMA model parameters
        for p in self.ema.parameters():
            p.requires_grad = False  # Use requires_grad property instead of requires_grad_()
            
        self.enabled = True

    def update(self, model):
        """Update EMA parameters."""
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if isinstance(v, jt.Var) and str(v.dtype).startswith(('float', 'half')):  # FP16 and FP32
                    v.update(v * d + (1 - d) * msd[k].detach())

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """Updates attributes and saves stripped model with optimizer removed."""
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


def strip_optimizer(f: Union[str, Path] = "best.pt", s: str = "", updates: dict = None) -> dict:
    """
    Strip optimizer from 'f' to finalize training, optionally save as 's'.

    Args:
        f (str): file path to model to strip the optimizer from. Default is 'best.pt'.
        s (str): file path to save the model with stripped optimizer to. If not provided, 'f' will be overwritten.
        updates (dict): a dictionary of updates to overlay onto the checkpoint before saving.

    Returns:
        (dict): The combined checkpoint dictionary.

    Example:
        ```python
        from pathlib import Path
        from jittoryolo.utils.torch_utils import strip_optimizer

        for f in Path("path/to/model/checkpoints").rglob("*.pt"):
            strip_optimizer(f)
        ```

    Note:
        Use `jittoryolo.nn.torch_safe_load` for missing modules with `x = torch_safe_load(f)[0]`
    """
    try:
        x = torch.load(f, map_location=torch.device("cpu"))
        assert isinstance(x, dict), "checkpoint is not a Python dictionary"
        assert "model" in x, "'model' missing from checkpoint"
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ Skipping {f}, not a valid jittoryolo model: {e}")
        return {}

    metadata = {
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://jittoryolo.com/license)",
        "docs": "https://docs.jittoryolo.com",
    }

    # Update model
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with EMA
    if hasattr(x["model"], "args"):
        x["model"].args = dict(x["model"].args)  # convert from IterableSimpleNamespace to dict
    if hasattr(x["model"], "criterion"):
        x["model"].criterion = None  # strip loss criterion
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False

    # Update other keys
    args = {**DEFAULT_CFG_DICT, **x.get("train_args", {})}  # combine args
    for k in "optimizer", "best_fitness", "ema", "updates":  # keys
        x[k] = None
    x["epoch"] = -1
    x["train_args"] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']

    # Save
    combined = {**metadata, **x, **(updates or {})}
    torch.save(combined, s or f)  # combine dicts (prefer to the right)
    mb = os.path.getsize(s or f) / 1e6  # file size
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")
    return combined


def convert_optimizer_state_dict_to_fp16(state_dict):
    """
    Converts the state_dict of a given optimizer to FP16, focusing on the 'state' key for tensor conversions.

    This method aims to reduce storage size without altering 'param_groups' as they contain non-tensor data.
    """
    for state in state_dict["state"].values():
        for k, v in state.items():
            if k != "step" and isinstance(v, torch.Tensor) and v.dtype is torch.float32:
                state[k] = v.half()

    return state_dict


def profile(input, ops, n=10, device=None):
    """
    jittoryolo speed, memory and FLOPs profiler.

    Example:
        ```python
        from jittoryolo.utils.torch_utils import profile

        input = torch.randn(16, 3, 640, 640)
        m1 = lambda x: x * torch.sigmoid(x)
        m2 = nn.SiLU()
        profile(input, [m1, m2], n=100)  # profile over 100 iterations
        ```
    """
    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)
    LOGGER.info(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}"
    )
    gc.collect()  # attempt to free unused memory
    torch.cuda.empty_cache()
    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, "to") else m  # device
            m = m.half() if hasattr(m, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=[x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
            except Exception:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception:  # no backward method
                        # print(e)  # for debug
                        t[2] = float("nan")
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else "list" for x in (x, y))  # shapes
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # parameters
                LOGGER.info(f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}")
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                LOGGER.info(e)
                results.append(None)
            finally:
                gc.collect()  # attempt to free unused memory
                torch.cuda.empty_cache()
    return results


class EarlyStopping:
    """Early stopping class that stops training when a specified number of epochs have passed without improvement."""

    def __init__(self, patience=50):
        """
        Initialize early stopping object.

        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping.
        """
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        """
        Check whether to stop training.

        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch

        Returns:
            (bool): True if training should stop, False otherwise
        """
        if fitness is None:  # check if fitness=None (happens when val=False)
            return False

        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            prefix = colorstr("EarlyStopping: ")
            LOGGER.info(
                f"{prefix}Training stopped early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `patience=300` or use `patience=0` to disable EarlyStopping."
            )
        return stop


class LambdaLR:
    """
    Custom LR scheduler that multiplies the learning rate by a given function.
    Implements similar functionality to PyTorch's LambdaLR.
    """
    def __init__(self, optimizer, lr_lambda):
        """Initialize LambdaLR scheduler."""
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.last_epoch = -1
        # Store initial learning rates from optimizer
        self.base_lrs = []
        for param_group in optimizer.param_groups:
            # In Jittor, we need to access the learning rate differently
            self.base_lrs.append(param_group.get("lr", param_group.get("learning_rate", 0.01)))
        
    def step(self):
        """Update learning rates for all parameter groups."""
        self.last_epoch += 1
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            # Calculate new learning rate
            new_lr = base_lr * self.lr_lambda(self.last_epoch)
            # Update learning rate in optimizer's param group
            # Some Jittor optimizers might use "learning_rate" instead of "lr"
            if "lr" in param_group:
                param_group["lr"] = new_lr
            else:
                param_group["learning_rate"] = new_lr

    def state_dict(self):
        """Returns scheduler state as a dictionary."""
        return {
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch
        }

    def load_state_dict(self, state_dict):
        """Loads scheduler state."""
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']

# Add lr_scheduler namespace to Jittor optim module if it doesn't exist
if not hasattr(jt.optim, 'lr_scheduler'):
    class LRScheduler:
        LambdaLR = LambdaLR
    
    jt.optim.lr_scheduler = LRScheduler
