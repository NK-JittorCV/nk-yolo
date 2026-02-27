# NK-YOLO 🚀 AGPL-3.0 License
# Refer to https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/jittor_utils.py

import math
import os
import random
import time
import importlib.util
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import jittor as jt
import jittor.nn as nn
import jittor.nn as F

from nkyolo.utils import (
    LOGGER,
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_KEYS,
    PYTHON_VERSION,
    RANK,
    __version__,
    colorstr,
)

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if _TORCH_AVAILABLE:
    import torch


@contextmanager
def autocast(enabled: bool, device: str = "cuda"):
    """Context manager for automatic mixed precision training.
    
    Args:
        enabled: Whether to enable AMP
        device: Device type ('cuda' or 'cpu')
    
    Note:
        For Jittor, AMP is handled internally. This context manager preserves
        the previous auto_mixed_precision_level setting to avoid precision changes
        between epochs.
    """
    # AMP is temporarily disabled.
    # prev_amp_level = jt.flags.auto_mixed_precision_level
    # if enabled:
    #     jt.flags.auto_mixed_precision_level = 1
    # else:
    #     jt.flags.auto_mixed_precision_level = 0
    # yield
    # jt.flags.auto_mixed_precision_level = prev_amp_level
    if enabled:
        raise NotImplementedError("AMP is temporarily disabled.")
    yield

def get_cpu_info():
    """Return a string with system CPU information, i.e. 'Apple M2'."""
    from nkyolo.utils import PERSISTENT_CACHE  # avoid circular import error

    if "cpu_info" not in PERSISTENT_CACHE:
        if importlib.util.find_spec("cpuinfo") is not None:
            import cpuinfo  # pip install py-cpuinfo

            k = "brand_raw", "hardware_raw", "arch_string_raw"  # keys sorted by preference
            info = cpuinfo.get_cpu_info()  # info dict
            key = k[0] if k[0] in info else k[1] if k[1] in info else k[2]
            string = info.get(key, "unknown")
            PERSISTENT_CACHE["cpu_info"] = string.replace("(R)", "").replace("CPU ", "").replace("@ ", "")
    return PERSISTENT_CACHE.get("cpu_info", "unknown")


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
    s = f"nkyolo {__version__} 🚀 Python-{PYTHON_VERSION} Jittor-{jt.__version__}"

    # Process device string
    device = str(device).lower().strip()
    # Remove unnecessary chars 
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")

    # MPI: do not remap CUDA_VISIBLE_DEVICES per-rank.
    # Let MPI/Jittor select GPU by mpi_local_rank within the visible list.
    _mpi_rank = os.getenv("OMPI_COMM_WORLD_RANK") or os.getenv("PMI_RANK") or os.getenv("RANK", "-1")
    _in_mpi = _mpi_rank != "-1"
    if _in_mpi:
        if device and device != "cpu":
            # Ignore explicit device in MPI to avoid mismatching local-rank vs visible devices.
            if verbose and RANK in {-1, 0}:
                LOGGER.warning("MPI detected: ignoring explicit device argument; use CUDA_VISIBLE_DEVICES via mpirun.")
            device = ""

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

    flops = get_flops(model, imgsz) if model.supports_flops else 0.0
    fused = " (fused)" if model.is_fused() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = model.yaml_file or model.yaml.get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    LOGGER.info(f"{model_name} summary{fused}: {n_l:,} layers, {n_p:,} parameters, {n_g:,} gradients{fs}")
    return n_l, n_p, n_g, flops


def get_num_params(model):
    """Return the total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())


def state_dict_to_numpy(state_dict, fp16=False):
    """Convert a state_dict of tensors to numpy arrays for portable checkpointing."""
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, jt.Var):
            arr = v.numpy()
        elif _TORCH_AVAILABLE and isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            arr = v
        else:
            out[k] = v
            continue
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        if fp16 and arr.dtype == np.float32:
            arr = arr.astype(np.float16)
        out[k] = arr
    return out


def state_dict_to_jittor(state_dict):
    """Convert a state_dict of numpy/torch tensors to Jittor tensors."""
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, jt.Var):
            out[k] = v
        elif isinstance(v, np.ndarray):
            if v.dtype in (np.float16, np.float64):
                v = v.astype(np.float32)
            out[k] = jt.array(v)
        elif _TORCH_AVAILABLE and isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
            if arr.dtype in (np.float16, np.float64):
                arr = arr.astype(np.float32)
            out[k] = jt.array(arr)
        else:
            out[k] = v
    return out


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
        from nkyolo.utils.benchmarks import ProfileModels

        results = ProfileModels([trainer.last], device=trainer.device).profile()[0]
        results.pop("model/name")
    else:  # only return PyTorch times from most recent validation
        results = {
            "model/parameters": get_num_params(trainer.model),
            "model/GFLOPs": round(get_flops(trainer.model), 3) if trainer.model.supports_flops else 0.0,
        }
    results["model/speed_PyTorch(ms)"] = round(trainer.validator.speed["inference"], 3)
    return results


def calculate_layer_flops(layer, input_shape):
    """Calculate FLOPs for a single layer (simplified estimation).
    
    Args:
        layer: Jittor layer/module
        input_shape: Input shape tuple (batch, channels, height, width)
    
    Returns:
        float: Estimated FLOPs (in GFLOPs)
    """
    from nkyolo.utils.jittor_profile import calculate_layer_flops as _calculate_layer_flops
    return _calculate_layer_flops(layer, input_shape) / 1e9  # Convert to GFLOPs


def get_flops(model, imgsz=640):
    """Return a YOLO model's FLOPs (GFLOPs)."""
    from nkyolo.utils.jittor_profile import profile

    model = de_parallel(model)
    disable_flops = str(os.getenv("NKYOLO_NO_FLOPS", "")).lower() in ("1", "true", "yes")
    force_flops = str(os.getenv("NKYOLO_FORCE_FLOPS", "")).lower() in ("1", "true", "yes")
    if disable_flops and not force_flops:
        return 0.0

    params = list(model.parameters())
    if len(params) == 0:
        return 0.0

    if not isinstance(imgsz, list):
        imgsz = [imgsz, imgsz]
    h, w = int(imgsz[0]), int(imgsz[1])

    # Resolve input channels.
    yaml = model.yaml if isinstance(getattr(model, "yaml", None), dict) else {}
    ch = yaml.get("ch", None)
    if isinstance(ch, (list, tuple)) and ch:
        ch = ch[0]
    if not isinstance(ch, (int, np.integer)) or ch <= 0:
        ch = None
    if ch is None:
        for m in model.modules():
            in_ch = getattr(m, "in_channels", None)
            if isinstance(in_ch, (int, np.integer)) and in_ch > 0:
                ch = int(in_ch)
                break
    if ch is None:
        for p_i in params:
            shape = getattr(p_i, "shape", None)
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                ch = int(shape[1])
                if ch > 0:
                    break
    if ch is None:
        ch = 3

    # Resolve profiling stride (same strategy as Ultralytics: fast then scale to target imgsz).
    stride = 32
    model_stride = getattr(model, "stride", None)
    try:
        if isinstance(model_stride, jt.Var):
            stride = max(int(model_stride.max().item()), 32)
        elif isinstance(model_stride, (list, tuple, np.ndarray)) and len(model_stride):
            stride = max(int(max(model_stride)), 32)
        elif isinstance(model_stride, (int, np.integer)) and model_stride > 0:
            stride = max(int(model_stride), 32)
    except Exception:
        stride = 32

    # Method 1: stride-based profiling and area scaling.
    try:
        im_stride = jt.empty(1, int(ch), int(stride), int(stride))
        flops_stride, _ = profile(model, inputs=[im_stride], verbose=False)  # raw FLOPs
        flops_stride_g = float(flops_stride) / 1e9
        if flops_stride_g > 0:
            return flops_stride_g * (h / stride) * (w / stride)
    except Exception:
        pass

    # Method 2: full-size fallback.
    try:
        im_full = jt.empty(1, int(ch), h, w)
        flops_full, _ = profile(model, inputs=[im_full], verbose=False)  # raw FLOPs
        return float(flops_full) / 1e9
    except Exception:
        return 0.0

def time_sync():
    """Return Jittor-accurate time."""
    if jt.flags.use_cuda:
        jt.sync_all(True)
    return time.time()

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
    w_bn = jt.diag(bn.weight.div(jt.sqrt(bn.eps + bn.running_var)))
    fuseddconv.weight.copy_(jt.matmul(w_bn, w_deconv).view(fuseddconv.weight.shape))

    # Prepare spatial bias
    b_conv = jt.zeros(deconv.weight.shape[1]) if deconv.bias is None else deconv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(jt.sqrt(bn.running_var + bn.eps))
    fuseddconv.bias.copy_(jt.matmul(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fuseddconv

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
        setattr(a, k, v)


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
    base_seed = int(seed)
    os.environ["NKYOLO_GLOBAL_SEED"] = str(base_seed)
    rank = int(jt.rank) if jt.in_mpi else 0
    if jt.in_mpi:
        seed = (base_seed + 1) * (rank + 1)
    else:
        seed = base_seed
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set Jittor's global seed.
    # Use the same seed across MPI ranks to keep model initialization identical.
    jt.set_global_seed(base_seed, different_seed_for_mpi=False)
    
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


def safe_deepcopy_jittor(obj):
    """
    Safe deepcopy function for Jittor objects that handles NanoVector and other non-picklable objects.
    
    Args:
        obj: Object to deepcopy
        
    Returns:
        Deep copy of the object, handling Jittor-specific objects
    """
    
    if isinstance(obj, jt.Var):
        return obj.clone()
    if isinstance(obj, dict):
        return {safe_deepcopy_jittor(k): safe_deepcopy_jittor(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_deepcopy_jittor(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(safe_deepcopy_jittor(v) for v in obj)
    if isinstance(obj, set):
        return {safe_deepcopy_jittor(v) for v in obj}
    return deepcopy(obj)


class ModelEMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models.
    Keeps a moving average of trainable parameters and copies buffers directly.
    """
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Initialize EMA for 'model' with given arguments."""
        # In MPI training, keep EMA on every rank so execution stays rank-consistent.
        in_mpi = bool(getattr(jt, "in_mpi", False))
        world_size = int(jt.world_size) if in_mpi else 1
        self.enabled = (RANK in (-1, 0)) if world_size <= 1 else True

        self.ema = None
        self.ema_state = {}
        self._ema_pairs = []
        self._ema_buffers = []
        self._backup_state = None
        self.updates = updates if self.enabled else 0  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp
        if not self.enabled:
            return

        model_sd = model.state_dict()
        skip_suffixes = (".anchors", ".strides")

        # Prefer a standalone EMA model so validation can use EMA weights
        # without swapping training model weights in-place.
        try:
            self.ema = deepcopy(model)
            for p in self.ema.parameters():
                if isinstance(p, jt.Var):
                    p.stop_grad()
        except Exception as e:
            self.ema = None
            if RANK in {-1, 0}:
                LOGGER.warning(f"WARNING ⚠️ EMA model deepcopy failed, fallback to state-only EMA: {e}")

        if self.ema is not None:
            ema_sd = self.ema.state_dict()
            for k, model_v in model_sd.items():
                if k.endswith(skip_suffixes) or k not in ema_sd:
                    continue
                ema_v = ema_sd[k]
                if isinstance(ema_v, jt.Var) and isinstance(model_v, jt.Var):
                    if model_v.is_stop_grad():
                        self._ema_buffers.append((ema_v, model_v))
                    else:
                        self._ema_pairs.append((ema_v, model_v))
            return

        # Fallback: keep EMA weights as state_dict-only tensors.
        for k, model_v in model_sd.items():
            if k.endswith(skip_suffixes):
                continue
            if isinstance(model_v, jt.Var):
                ema_v = model_v.clone()
                if model_v.is_stop_grad():
                    self._ema_buffers.append((ema_v, model_v))
                else:
                    self._ema_pairs.append((ema_v, model_v))
            else:
                ema_v = deepcopy(model_v)
            self.ema_state[k] = ema_v

    def update(self, model, updates=None):
        """Update EMA parameters."""
        if not self.enabled:
            return
        if self.ema is None and not self.ema_state:
            return
        if updates is None:
            self.updates += 1
        else:
            self.updates = int(updates)
        with jt.no_grad():
            d = self.decay(self.updates) if callable(self.decay) else self.decay
            for ema_v, model_v in self._ema_pairs:
                ema_v.update(ema_v * d + (1 - d) * model_v)
            for ema_v, model_v in self._ema_buffers:
                ema_v.update(model_v)
            if jt.flags.use_cuda and not jt.in_mpi:
                jt.sync_all(True)

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """Updates attributes and saves stripped model with optimizer removed."""
        if self.enabled and self.ema is not None:
            copy_attr(self.ema, model, include, exclude)

    def state_dict(self):
        if self.ema is not None:
            return self.ema.state_dict()
        return self.ema_state

    def load_state_dict(self, state_dict):
        if not self.enabled or not state_dict:
            return
        if self.ema is not None:
            self.ema.load_state_dict(state_dict)
            return
        for k, v in state_dict.items():
            if k not in self.ema_state:
                continue
            ema_v = self.ema_state[k]
            if isinstance(ema_v, jt.Var) and isinstance(v, jt.Var):
                ema_v.update(v)
            else:
                self.ema_state[k] = v

    def apply_to(self, model):
        if not self.enabled or self.ema is not None or not self.ema_state:
            return False
        self._backup_state = safe_deepcopy_jittor(model.state_dict())
        model.load_state_dict(self.ema_state)
        return True

    def restore(self, model):
        if self._backup_state is None:
            return
        model.load_state_dict(self._backup_state)
        self._backup_state = None



def strip_optimizer(f: Union[str, Path] = "best.pkl", s: str = "", updates: dict = None) -> dict:
    """
    Strip optimizer from 'f' to finalize training, optionally save as 's'.

    Args:
        f (str): file path to model to strip the optimizer from. Default is 'best.pkl'.
        s (str): file path to save the model with stripped optimizer to. If not provided, 'f' will be overwritten.
        updates (dict): a dictionary of updates to overlay onto the checkpoint before saving.

    Returns:
        (dict): The combined checkpoint dictionary.

    Example:
        ```python
        from pathlib import Path
        from nkyolo.utils.torch_utils import strip_optimizer

        for f in Path("path/to/model/checkpoints").rglob("*.pkl"):
            strip_optimizer(f)
        ```

    Note:
        Use `nkyolo.nn.torch_safe_load` for missing modules with `x = torch_safe_load(f)[0]`
    """
    x = jt.load(str(f))
    if not isinstance(x, dict):
        raise TypeError(f"Checkpoint '{f}' is not a dictionary.")
    if "model" not in x and "ema" not in x:
        raise KeyError(f"Checkpoint '{f}' has no model/ema weights.")

    metadata = {
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }

    # Update model weights
    model_weights = x.get("ema") or x.get("model")
    if isinstance(model_weights, nn.Module):
        model_weights = model_weights.state_dict()
    x["model"] = state_dict_to_numpy(model_weights, fp16=True)

    # Update other keys
    args = {**DEFAULT_CFG_DICT, **x.get("train_args", {})}  # combine args
    for k in "optimizer", "best_fitness", "ema", "updates":  # keys
        x[k] = None
    x["epoch"] = -1
    x["train_args"] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']

    # Save
    combined = {**metadata, **x, **(updates or {})}
    jt.save(combined, str(s) or str(f))  # combine dicts (prefer to the right)
    mb = os.path.getsize(str(s) or  str(f)) / 1e6  # file size
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")
    return combined


def convert_optimizer_state_dict_to_fp16(state_dict):
    """
    Converts the state_dict of a given optimizer to FP16, handling both PyTorch and Jittor optimizers.
    
    This method aims to reduce storage size without altering 'param_groups' as they contain non-tensor data.
    """
    # Handle PyTorch-style optimizers with 'state' key
    if "state" in state_dict:
        for state in state_dict["state"].values():
            for k, v in state.items():
                if k != "step" and isinstance(v, jt.Var) and str(v.dtype).startswith("float32"):
                    state[k] = v.half()
    
    # Handle Jittor optimizers that have 'defaults' key instead of 'state'
    # Jittor optimizers typically don't have tensor states that need conversion
    # They store optimizer parameters in 'defaults' which are mostly scalars
    elif "defaults" in state_dict:
        # For Jittor optimizers, we don't need to convert anything to FP16
        # as they don't store tensor states like PyTorch optimizers
        pass
    
    return state_dict


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

def profile(input, ops, n=10, device=None):
    """
    NK-YOLO speed, memory and FLOPs profiler for Jittor.

    Example:
        ```python
        from nkyolo.utils.jittor_utils import profile

        input = jt.randn(16, 3, 640, 640)
        m1 = lambda x: x * jt.sigmoid(x)
        m2 = nn.SiLU()
        profile(input, [m1, m2], n=100)  # profile over 100 iterations
        ```
    """
    results = []
    LOGGER.info(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}"
    )

    for x in input if isinstance(input, list) else [input]:
        if not isinstance(x, jt.Var):
            x = jt.array(x)
        if device is not None:
            if isinstance(device, str):
                if device == "cuda":
                    x = x.cuda()
                elif device == "cpu":
                    x = x.cpu()
            else:
                x = x.to(device)

        x.start_grad()

        for m in ops if isinstance(ops, list) else [ops]:
            if not isinstance(m, nn.Module):
                raise TypeError("profile() expects nn.Module instances.")
            if device is not None:
                m = m.to(device)
            if x.dtype == jt.float16:
                m = m.half()

            from nkyolo.utils.jittor_profile import profile as unified_profile

            flops_result, _ = unified_profile(m, inputs=[x], verbose=False)
            flops = flops_result / 1e9

            tf = tb = 0.0
            for _ in range(n):
                t0 = time_sync()
                y = m(x)
                t1 = time_sync()
                loss = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum()
                loss.backward()
                t2 = time_sync()
                tf += (t1 - t0) * 1000 / n
                tb += (t2 - t1) * 1000 / n

            s_in = tuple(x.shape)
            s_out = tuple(y.shape) if isinstance(y, jt.Var) else ("list" if isinstance(y, list) else str(type(y)))
            p = sum(v.numel() for v in m.parameters())
            mem = 0

            LOGGER.info(f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}")
            results.append([p, flops, mem, tf, tb, s_in, s_out])
    return results
