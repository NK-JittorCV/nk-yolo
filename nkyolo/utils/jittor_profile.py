# NK-YOLO 🚀 AGPL-3.0 License
# Unified FLOPs profile tool for Jittor models.

import jittor as jt
import jittor.nn as nn

from nkyolo.utils import LOGGER


def _to_2tuple(v, default=1):
    """Convert scalar/list/tuple to 2-tuple of ints."""
    if isinstance(v, (tuple, list)):
        if len(v) >= 2:
            return int(v[0]), int(v[1])
        if len(v) == 1:
            return int(v[0]), int(v[0])
        return int(default), int(default)
    return int(v), int(v)


def _numel_shape(shape):
    """Return number of elements from a shape-like sequence."""
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def _first_var(x):
    """Get first Jittor Var from nested input/output structures."""
    if isinstance(x, jt.Var):
        return x
    if isinstance(x, (list, tuple)):
        for xi in x:
            yi = _first_var(xi)
            if yi is not None:
                return yi
    return None


def _to_var_tree(x):
    """Convert ndarray/list tensor inputs to Jittor Vars while preserving nesting."""
    if isinstance(x, jt.Var):
        return x
    if isinstance(x, (str, bytes, bool, int, float, type(None))):
        return x
    if isinstance(x, dict):
        return {k: _to_var_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_var_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_var_tree(v) for v in x)
    return jt.array(x)


def _normalize_inputs(inputs):
    """Normalize model inputs to a list."""
    if not isinstance(inputs, list):
        inputs = [inputs]
    return [_to_var_tree(x) for x in inputs]


def _run_module(module, inputs):
    """Run a module with list-formatted inputs."""
    if len(inputs) == 1:
        return module(inputs[0])
    return module(*inputs)


def _is_leaf_module(module):
    """A leaf module has no children."""
    return len(list(module.children())) == 0


def _conv2d_output_hw(input_shape, layer):
    """Infer Conv2d output spatial size from input shape and layer attrs."""
    if len(input_shape) < 4:
        return None
    in_h, in_w = int(input_shape[2]), int(input_shape[3])
    k_h, k_w = _to_2tuple(getattr(layer, "kernel_size", 1))
    s_h, s_w = _to_2tuple(getattr(layer, "stride", 1))
    p_h, p_w = _to_2tuple(getattr(layer, "padding", 0))
    d_h, d_w = _to_2tuple(getattr(layer, "dilation", 1))

    out_h = (in_h + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
    out_w = (in_w + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1
    return max(int(out_h), 0), max(int(out_w), 0)


def _conv_transpose_output_hw(input_shape, layer):
    """Infer ConvTranspose output spatial size from input shape and layer attrs."""
    if len(input_shape) < 4:
        return None
    in_h, in_w = int(input_shape[2]), int(input_shape[3])
    k_h, k_w = _to_2tuple(getattr(layer, "kernel_size", 1))
    s_h, s_w = _to_2tuple(getattr(layer, "stride", 1))
    p_h, p_w = _to_2tuple(getattr(layer, "padding", 0))
    d_h, d_w = _to_2tuple(getattr(layer, "dilation", 1))
    op_h, op_w = _to_2tuple(getattr(layer, "output_padding", 0), default=0)

    out_h = (in_h - 1) * s_h - 2 * p_h + d_h * (k_h - 1) + op_h + 1
    out_w = (in_w - 1) * s_w - 2 * p_w + d_w * (k_w - 1) + op_w + 1
    return max(int(out_h), 0), max(int(out_w), 0)


def calculate_layer_flops(layer, input_shape):
    """Estimate FLOPs for a single layer from input shape (raw FLOPs count)."""
    if not isinstance(layer, nn.Module):
        return 0.0

    if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 and isinstance(input_shape[0], (list, tuple)):
        # For list-of-shapes inputs, use the first tensor shape.
        input_shape = input_shape[0]

    if not isinstance(input_shape, (list, tuple)) or len(input_shape) == 0:
        return 0.0

    input_shape = tuple(int(x) for x in input_shape)
    layer_type = type(layer).__name__.lower()

    # Conv2d-like
    if "convtranspose" in layer_type:
        if len(input_shape) < 4:
            return 0.0
        out_hw = _conv_transpose_output_hw(input_shape, layer)
        if out_hw is None:
            return 0.0
        b = int(input_shape[0])
        out_c = int(getattr(layer, "out_channels", 0))
        if out_c <= 0:
            return 0.0
        in_c = int(getattr(layer, "in_channels", input_shape[1]))
        groups = max(int(getattr(layer, "groups", 1) or 1), 1)
        k_h, k_w = _to_2tuple(getattr(layer, "kernel_size", 1))
        out_h, out_w = out_hw
        out_elements = b * out_c * out_h * out_w
        macs = out_elements * (in_c / groups) * k_h * k_w
        if getattr(layer, "bias", None) is not None:
            macs += out_elements
        return float(macs * 2)

    if layer_type in {"conv", "conv2d"}:
        if len(input_shape) < 4:
            return 0.0
        out_hw = _conv2d_output_hw(input_shape, layer)
        if out_hw is None:
            return 0.0
        b = int(input_shape[0])
        out_c = int(getattr(layer, "out_channels", 0))
        if out_c <= 0:
            return 0.0
        in_c = int(getattr(layer, "in_channels", input_shape[1]))
        groups = max(int(getattr(layer, "groups", 1) or 1), 1)
        k_h, k_w = _to_2tuple(getattr(layer, "kernel_size", 1))
        out_h, out_w = out_hw
        out_elements = b * out_c * out_h * out_w
        macs = out_elements * (in_c / groups) * k_h * k_w
        if getattr(layer, "bias", None) is not None:
            macs += out_elements
        return float(macs * 2)

    # Linear
    if "linear" in layer_type:
        in_features = int(getattr(layer, "in_features", input_shape[-1]))
        if in_features <= 0:
            return 0.0
        out_features = int(getattr(layer, "out_features", 0))
        if out_features <= 0:
            return 0.0
        batch = _numel_shape(input_shape[:-1]) if len(input_shape) > 1 else int(input_shape[0])
        out_elements = batch * out_features
        macs = out_elements * in_features
        if getattr(layer, "bias", None) is not None:
            macs += out_elements
        return float(macs * 2)

    # Norm layers (inference path)
    if "batchnorm" in layer_type or "layernorm" in layer_type:
        return float(_numel_shape(input_shape) * 4)

    # Activations
    if layer_type in {"relu", "relu6", "leakyrelu", "leaky_relu", "hardtanh"}:
        return float(_numel_shape(input_shape))
    if layer_type in {"silu", "sigmoid", "tanh", "gelu", "mish", "hardsigmoid", "hardswish"}:
        return float(_numel_shape(input_shape) * 4)

    # Pooling (rough estimate)
    if "pool" in layer_type and len(input_shape) >= 4:
        k = getattr(layer, "kernel_size", None)
        if k is None and hasattr(layer, "_layer"):
            k = getattr(layer._layer, "kernel_size", None)
        if k is None and hasattr(layer, "layer"):
            k = getattr(layer.layer, "kernel_size", None)
        k_h, k_w = _to_2tuple(k, default=1)
        # Roughly use input elements scaled by stride area.
        s = getattr(layer, "stride", None)
        if s is None and hasattr(layer, "_layer"):
            s = getattr(layer._layer, "stride", None)
        if s is None and hasattr(layer, "layer"):
            s = getattr(layer.layer, "stride", None)
        s_h, s_w = _to_2tuple(s, default=1)
        out_h = max(int(input_shape[2] // max(s_h, 1)), 1)
        out_w = max(int(input_shape[3] // max(s_w, 1)), 1)
        out_elements = int(input_shape[0]) * int(input_shape[1]) * out_h * out_w
        return float(out_elements * k_h * k_w)

    # Upsample/Interpolate
    if "upsample" in layer_type or "interpolate" in layer_type:
        scale = getattr(layer, "scale_factor", 1.0)
        if isinstance(scale, (list, tuple)):
            s_h, s_w = float(scale[0]), float(scale[1])
        else:
            s_h = s_w = float(scale)
        if len(input_shape) >= 4:
            out_elements = int(input_shape[0]) * int(input_shape[1]) * int(input_shape[2] * s_h) * int(input_shape[3] * s_w)
            return float(out_elements)
        return 0.0

    # Concat/Add/etc. are small compared to convs, default to 0 here.
    return 0.0


def _calculate_flops_from_io(layer, inputs, outputs):
    """Estimate FLOPs for one executed layer from concrete input/output tensors."""
    if not isinstance(layer, nn.Module):
        return 0.0

    x = _first_var(inputs)
    y = _first_var(outputs)
    if x is None or y is None:
        return 0.0

    x_shape = tuple(int(d) for d in x.shape)
    y_shape = tuple(int(d) for d in y.shape)
    if len(y_shape) == 0:
        return 0.0

    layer_type = type(layer).__name__.lower()
    out_elements = _numel_shape(y_shape)

    # Conv / ConvTranspose
    if "convtranspose" in layer_type or layer_type in {"conv", "conv2d"}:
        if len(y_shape) < 4:
            return 0.0
        in_c = int(getattr(layer, "in_channels", x_shape[1] if len(x_shape) > 1 else 0))
        groups = max(int(getattr(layer, "groups", 1) or 1), 1)
        k_h, k_w = _to_2tuple(getattr(layer, "kernel_size", 1))
        macs = out_elements * (in_c / groups) * k_h * k_w
        if getattr(layer, "bias", None) is not None:
            macs += out_elements
        return float(macs * 2)

    # Linear
    if "linear" in layer_type:
        in_features = int(getattr(layer, "in_features", x_shape[-1] if len(x_shape) else 0))
        if in_features <= 0:
            return 0.0
        macs = out_elements * in_features
        if getattr(layer, "bias", None) is not None:
            macs += out_elements
        return float(macs * 2)

    # Normalization
    if "batchnorm" in layer_type or "layernorm" in layer_type:
        return float(out_elements * 4)

    # Activations
    if layer_type in {"relu", "relu6", "leakyrelu", "leaky_relu", "hardtanh"}:
        return float(out_elements)
    if layer_type in {"silu", "sigmoid", "tanh", "gelu", "mish", "hardsigmoid", "hardswish"}:
        return float(out_elements * 4)
    if "softmax" in layer_type:
        return float(out_elements * 5)

    # Pooling
    if "pool" in layer_type and len(y_shape) >= 4:
        k = getattr(layer, "kernel_size", None)
        if k is None and hasattr(layer, "_layer"):
            k = getattr(layer._layer, "kernel_size", None)
        if k is None and hasattr(layer, "layer"):
            k = getattr(layer.layer, "kernel_size", None)
        if k is None:
            # Adaptive pooling fallback.
            if len(x_shape) >= 4 and y_shape[2] > 0 and y_shape[3] > 0:
                k_h = max(int(round(x_shape[2] / y_shape[2])), 1)
                k_w = max(int(round(x_shape[3] / y_shape[3])), 1)
            else:
                k_h = k_w = 1
        else:
            k_h, k_w = _to_2tuple(k, default=1)
        return float(out_elements * k_h * k_w)

    # Upsample/Interpolate
    if "upsample" in layer_type or "interpolate" in layer_type:
        mode = str(getattr(layer, "mode", "nearest")).lower()
        if "nearest" in mode:
            return float(out_elements)
        # bilinear/bicubic: rough estimate
        return float(out_elements * 4)

    # Concat/add/mul and custom wrappers are intentionally ignored here.
    return 0.0


def _profile_with_hooks(model, inputs, verbose=False):
    """Profile FLOPs by attaching temporary forward hooks on leaf modules."""
    if not isinstance(model, nn.Module):
        return 0.0, 0.0

    inputs = _normalize_inputs(inputs)
    total_flops = 0.0
    total_params = float(sum(int(p.numel()) for p in model.parameters()))
    layer_logs = []

    leaf_modules = [m for m in model.modules() if _is_leaf_module(m)]
    prev_hooks = {}

    def _hook(mod, inps, outs):
        nonlocal total_flops
        layer_flops = _calculate_flops_from_io(mod, inps, outs)
        total_flops += layer_flops
        if verbose:
            layer_params = sum(int(p.numel()) for p in mod.parameters())
            layer_logs.append((type(mod).__name__, layer_flops, layer_params))

    for mod in leaf_modules:
        prev_hooks[mod] = getattr(mod, "__fhook__", None)
        mod.register_forward_hook(_hook)

    model_was_training = model.training
    model.eval()
    try:
        with jt.no_grad():
            out = _run_module(model, inputs)
            if isinstance(out, jt.Var):
                out.sync()
            else:
                jt.sync_all(True)
    finally:
        for mod, prev_hook in prev_hooks.items():
            if prev_hook is None:
                mod.remove_forward_hook()
            else:
                mod.register_forward_hook(prev_hook)
        if model_was_training:
            model.train()
        else:
            model.eval()

    if verbose:
        for layer_name, layer_flops, layer_params in layer_logs:
            LOGGER.info(f"{layer_name:<45} {layer_flops / 1e9:10.4f} GFLOPs, {layer_params:12d} params")

    return float(total_flops), total_params


def profile_model_recursive(model, inputs, verbose=False):
    """Backward-compatible API: profile model FLOPs/params (raw FLOPs count)."""
    return _profile_with_hooks(model, inputs, verbose=verbose)


def profile_jittor(model, inputs, verbose=False):
    """Profile a Jittor model and return (flops_raw, params)."""
    return _profile_with_hooks(model, inputs, verbose=verbose)


def _infer_input_shape(x):
    """Infer a representative tensor shape from nested input."""
    v = _first_var(x)
    if v is not None:
        return tuple(int(d) for d in v.shape)
    if isinstance(x, (list, tuple)):
        for item in x:
            if isinstance(item, (list, tuple)):
                s = _infer_input_shape(item)
                if len(s):
                    return s
    return (1,)


def profile_model_graph(model, inputs, verbose=False):
    """Backward-compatible API: graph-aware profiling now uses hook-based profiling."""
    return _profile_with_hooks(model, inputs, verbose=verbose)


def profile(model, inputs, verbose=False, *args, **kwargs):
    """Profile a Jittor model using the native hook-based implementation."""
    return profile_jittor(model, inputs, verbose=verbose)


def profile_layer_with_fallback(layer, inputs):
    """Profile a layer in GFLOPs with fallback to shape-based estimation."""
    if not isinstance(layer, nn.Module):
        return 0.0

    try:
        flops, _ = _profile_with_hooks(layer, inputs, verbose=False)
        if flops > 0:
            return flops / 1e9
    except Exception:
        pass

    inputs_list = inputs if isinstance(inputs, list) else [inputs]
    input_shape = _infer_input_shape(inputs_list[0]) if len(inputs_list) else (1,)
    return calculate_layer_flops(layer, input_shape) / 1e9


# For backward compatibility and human-friendly formatting
def clever_format(nums, format="%.2f"):
    """Format numbers in a human-readable way."""
    if not isinstance(nums, (list, tuple)):
        nums = [nums]

    result = []
    for num in nums:
        if num >= 1e12:
            result.append(f"{format} T" % (num / 1e12))
        elif num >= 1e9:
            result.append(f"{format} G" % (num / 1e9))
        elif num >= 1e6:
            result.append(f"{format} M" % (num / 1e6))
        elif num >= 1e3:
            result.append(f"{format} K" % (num / 1e3))
        else:
            result.append(f"{format}" % num)

    return result[0] if len(result) == 1 else result
