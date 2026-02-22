# NK-YOLO 🚀 AGPL-3.0 License
# Unified profile tool for Jittor models (native implementation).

import jittor as jt
import jittor.nn as nn

from nkyolo.utils import LOGGER


def calculate_layer_flops(layer, input_shape):
    """Calculate FLOPs for a single layer.
    
    Args:
        layer: Jittor layer/module
        input_shape: Input shape tuple (batch, channels, height, width) or (batch, features)
    
    Returns:
        float: Estimated FLOPs (not converted to GFLOPs, returns raw count)
    """
    if not isinstance(layer, nn.Module):
        return 0.0
    
    layer_type = type(layer).__name__
    flops = 0.0
    
    if layer_type == "Conv2d":
        # FLOPs = (kernel_h * kernel_w * in_channels * out_channels + out_channels) * output_h * output_w
        out_channels = layer.out_channels
        in_channels = layer.in_channels
        kernel_size = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
        k_h, k_w = kernel_size
        
        # Calculate output size
        if len(input_shape) >= 4:
            output_h, output_w = input_shape[2], input_shape[3]
            padding = layer.padding
            if isinstance(padding, int):
                padding = (padding, padding)
            elif isinstance(padding, tuple) and len(padding) == 1:
                padding = (padding[0], padding[0])
            else:
                padding = padding if isinstance(padding, tuple) else (0, 0)

            stride = layer.stride
            if isinstance(stride, int):
                stride = (stride, stride)
            elif isinstance(stride, tuple) and len(stride) == 1:
                stride = (stride[0], stride[0])
            else:
                stride = stride if isinstance(stride, tuple) else (1, 1)

            output_h = (input_shape[2] + 2 * padding[0] - k_h) // stride[0] + 1
            output_w = (input_shape[3] + 2 * padding[1] - k_w) // stride[1] + 1
            
            # MACs = kernel_size * in_channels * out_channels * output_size
            # FLOPs = MACs * 2 (multiply-add operations)
            macs = k_h * k_w * in_channels * out_channels * output_h * output_w
            # Add bias operations if exists
            if layer.bias is not None:
                macs += out_channels * output_h * output_w
            flops = macs * 2  # Multiply-add counts as 2 operations
    
    elif layer_type == "Linear":
        # FLOPs = (in_features * out_features + out_features) * batch_size
        in_features = layer.in_features
        out_features = layer.out_features
        batch_size = input_shape[0] if len(input_shape) > 1 else 1
        
        macs = in_features * out_features * batch_size
        if layer.bias is not None:
            macs += out_features * batch_size
        flops = macs * 2
    
    elif layer_type == "BatchNorm2d":
        # FLOPs = channels * height * width * 4 (mean + variance + normalize + scale_shift)
        if len(input_shape) >= 4:
            flops = input_shape[1] * input_shape[2] * input_shape[3] * 4
    
    elif layer_type == "LayerNorm":
        # FLOPs = features * 4 (similar to BatchNorm)
        if len(input_shape) >= 2:
            features = input_shape[-1]
            batch_size = input_shape[0] if len(input_shape) > 1 else 1
            flops = features * batch_size * 4
    
    elif layer_type == "ReLU":
        # FLOPs = number of elements (simple comparison operation)
        total_elements = 1
        for dim in input_shape:
            total_elements *= dim
        flops = total_elements
    
    elif layer_type in ["SiLU", "Sigmoid", "Tanh"]:
        # FLOPs = elements * 3 (exp/log/div operations)
        total_elements = 1
        for dim in input_shape:
            total_elements *= dim
        flops = total_elements * 3
    
    elif layer_type == "MaxPool2d" or layer_type == "AvgPool2d":
        # FLOPs = kernel_size * output_size (for pooling operations)
        if len(input_shape) >= 4:
            pool = layer._layer if layer_type == "MaxPool2d" else layer.layer
            kernel_size = pool.kernel_size
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            
            stride = pool.stride
            if isinstance(stride, int):
                stride = (stride, stride)
            
            padding = pool.padding
            if isinstance(padding, int):
                padding = (padding, padding)
            
            output_h = (input_shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
            output_w = (input_shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
            
            total_elements = input_shape[0] * input_shape[1] * output_h * output_w
            flops = total_elements * kernel_size[0] * kernel_size[1]
    
    elif layer_type == "Upsample" or layer_type == "Interpolate":
        # FLOPs = output_size (simple copying/interpolation)
        if len(input_shape) >= 4:
            scale_factor = layer.scale_factor
            if isinstance(scale_factor, (int, float)):
                scale_factor = (scale_factor, scale_factor)
            
            output_h = int(input_shape[2] * scale_factor[0])
            output_w = int(input_shape[3] * scale_factor[1])
            flops = input_shape[0] * input_shape[1] * output_h * output_w
    
    elif layer_type == "Concat" or layer_type == "Cat":
        # Concatenation has minimal FLOPs (just memory copy)
        flops = 0
    
    elif layer_type == "Add" or layer_type == "Multiply":
        # Element-wise operations: FLOPs = number of elements
        total_elements = 1
        for dim in input_shape:
            total_elements *= dim
        flops = total_elements

    return float(flops)


def profile_model_recursive(model, inputs, verbose=False):
    """Recursively calculate FLOPs for all layers in a model.
    
    Args:
        model: Jittor model/module
        inputs: Input tensor or list of input tensors
        verbose: Whether to print detailed information
    
    Returns:
        tuple: (total_flops, total_params) where flops is in raw count (not GFLOPs)
    """
    total_flops = 0.0
    total_params = 0.0

    if not isinstance(model, nn.Module):
        return total_flops, total_params
    
    # Convert inputs to list if needed
    if not isinstance(inputs, list):
        inputs_list = [inputs]
    else:
        inputs_list = inputs
    
    # Get input tensor
    x = inputs_list[0]
    if not isinstance(x, jt.Var):
        x = jt.array(x)
    
    # Store intermediate outputs for shape tracking
    layer_shapes = {}
    layer_shapes['input'] = tuple(x.shape)
    
    # Manually traverse model modules and calculate FLOPs
    def traverse_and_calculate(module, x, prefix=''):
        nonlocal total_flops
        current_x = x
        
        # Get all child modules
        children = list(module.named_children())

        if len(children) == 0 or not isinstance(module, nn.Sequential):
            # Leaf or non-sequential composite module - calculate FLOPs directly
            input_shape = tuple(current_x.shape) if isinstance(current_x, jt.Var) else _infer_input_shape(current_x)
            layer_flops = calculate_layer_flops(module, input_shape)
            total_flops += layer_flops

            if verbose:
                params = sum(p.numel() for p in module.parameters())
                layer_name = prefix if prefix else type(module).__name__
                print(f"{layer_name:50s} {layer_flops/1e9:12.4f} GFLOPs, {params:12d} params")

            # Try to get output shape
            with jt.no_grad():
                output = module(current_x)
                if isinstance(output, jt.Var):
                    return output
            return current_x

        # Sequential container module - traverse children in order
        for name, child in children:
            full_name = f"{prefix}.{name}" if prefix else name
            current_x = traverse_and_calculate(child, current_x, prefix=full_name)
        return current_x
    
    # Set model to eval mode for profiling
    model_was_training = model.training
    model.eval()
    
    with jt.no_grad():
        _ = traverse_and_calculate(model, x)
    if model_was_training:
        model.train()
    else:
        model.eval()

    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    return total_flops, total_params


def profile_jittor(model, inputs, verbose=False):
    """Profile a Jittor model to calculate FLOPs and parameters.
    
    Args:
        model: Jittor model/module
        inputs: Input tensor or list of input tensors
        verbose: Whether to print detailed information
    
    Returns:
        tuple: (flops, params) where:
            - flops: Total FLOPs count (raw number, not divided by 1e9)
            - params: Total number of parameters
    """
    # Handle inputs format (expects inputs as list)
    if not isinstance(inputs, list):
        inputs = [inputs]
    
    # Ensure inputs are Jittor Vars
    inputs = [jt.array(inp) if not isinstance(inp, jt.Var) else inp for inp in inputs]
    
    # Calculate FLOPs and params
    flops, params = profile_model_recursive(model, inputs, verbose=verbose)
    
    return (flops, params)


def _infer_input_shape(x):
    """Infer a representative input shape from a tensor or list/tuple of tensors."""
    if isinstance(x, jt.Var):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        for item in x:
            if isinstance(item, jt.Var):
                return tuple(item.shape)
    return (1,)


def profile_model_graph(model, inputs, verbose=False):
    """Profile a model with YOLO-style graph routing using module .f and saved outputs."""
    if not isinstance(inputs, list):
        inputs = [inputs]
    x = inputs[0]
    if not isinstance(x, jt.Var):
        x = jt.array(x)

    total_flops = 0.0
    total_params = sum(p.numel() for p in model.parameters())

    y = []
    save = model.save
    model_was_training = model.training
    model.eval()
    with jt.no_grad():
        for m in model.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            layer_flops = 0.0
            if isinstance(x, (list, tuple)):
                m_type = type(m).__name__
                if m_type in {"Detect", "Segment", "Pose", "OBB", "WorldDetect", "v10Detect"}:
                    for xi, cv2_i, cv3_i in zip(x, m.cv2, m.cv3):
                        layer_flops += profile_model_recursive(cv2_i, inputs=[xi], verbose=False)[0]
                        layer_flops += profile_model_recursive(cv3_i, inputs=[xi], verbose=False)[0]
                    if m_type in {"Segment", "Pose", "OBB"}:
                        for xi, cv4_i in zip(x, m.cv4):
                            layer_flops += profile_model_recursive(cv4_i, inputs=[xi], verbose=False)[0]
                    if m_type == "Segment":
                        layer_flops += profile_model_recursive(m.proto, inputs=[x[0]], verbose=False)[0]
                total_flops += layer_flops
            else:
                input_shape = _infer_input_shape(x)
            layer_flops = calculate_layer_flops(m, input_shape)
            if layer_flops == 0.0 and isinstance(m, nn.Sequential):
                children = list(m.named_children())
                if children:
                    layer_flops = profile_model_recursive(m, inputs=[x], verbose=False)[0]
            total_flops += layer_flops
            x = m(x)
            y.append(x if m.i in save else None)
            if verbose:
                LOGGER.info(f"{m.type:<45} {total_flops / 1e9:10.4f} GFLOPs")
    if model_was_training:
        model.train()
    else:
        model.eval()
    return total_flops, total_params


def profile(model, inputs, verbose=False, *args, **kwargs):
    """Profile a Jittor model using the native profiling implementation.
    
    Args:
        model: Jittor model/module
        inputs: Input tensor(s) or list of input tensors
        verbose: Whether to print detailed information
        *args: Unused (kept for compatibility)
        **kwargs: Unused (kept for compatibility)
    
    Returns:
        tuple: (flops, params) where:
            - flops: Total FLOPs count (raw number, not divided by 1e9)
            - params: Total number of parameters
    
    Example:
        ```python
        # Jittor example
        import jittor as jt
        import jittor.nn as nn
        from nkyolo.utils.jittor_profile import profile
        
        model = nn.Conv2d(3, 64, 3, padding=1)
        x = jt.randn(1, 3, 224, 224)
        flops, params = profile(model, inputs=[x], verbose=True)
        ```
    """
    return profile_jittor(model, inputs, verbose=verbose)


def profile_layer_with_fallback(layer, inputs):
    """Profile a single layer with a lightweight fallback calculation.
    
    Args:
        layer: Layer/module to profile (PyTorch or Jittor)
        inputs: Input tensor(s) or list of input tensors
    
    Returns:
        float: FLOPs in GFLOPs (already divided by 1e9)
    """
    # Normalize inputs to get input shape
    if not isinstance(inputs, list):
        inputs_list = [inputs]
    else:
        inputs_list = inputs

    if len(inputs_list) > 0:
        x = inputs_list[0]
        input_shape = tuple(x.shape) if isinstance(x, jt.Var) else (1,)
    else:
        input_shape = (1,)

    # Check if layer is a valid module
    if isinstance(layer, nn.Module):
        flops = calculate_layer_flops(layer, input_shape)
        return flops / 1e9  # Convert to GFLOPs
    return 0.0


# For backward compatibility and human-friendly formatting
def clever_format(nums, format="%.2f"):
    """Format numbers in a human-readable way.
    
    Args:
        nums: Single number or list of numbers
        format: Format string
    
    Returns:
        Formatted string or list of formatted strings
    """
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
