# NK-YOLO 🚀 AGPL-3.0 License
# Unified profile tool that supports both PyTorch (via thop) and Jittor (native)
#
# This module provides a unified interface that automatically detects whether
# the model and inputs are PyTorch or Jittor, and uses the appropriate profiling tool:
# - PyTorch: uses thop.profile (requires thop package)
# - Jittor: uses native Jittor profile implementation

from importlib.util import find_spec

import jittor as jt
import jittor.nn as nn

# Check module availability using importlib instead of exception handling
def _module_available(module_name):
    """Check if a module is available without importing it."""
    return find_spec(module_name) is not None

TORCH_AVAILABLE = _module_available("torch")
THOP_AVAILABLE = _module_available("thop")

# Lazy import torch and thop only when needed
_torch = None
_thop = None

def _get_torch():
    """Lazy import torch module."""
    global _torch
    if _torch is None and TORCH_AVAILABLE:
        import torch
        _torch = torch
    return _torch

def _get_thop():
    """Lazy import thop module."""
    global _thop
    if _thop is None and THOP_AVAILABLE:
        import thop
        _thop = thop
    return _thop


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
            if hasattr(layer, 'padding'):
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
            if hasattr(layer, 'bias') and layer.bias is not None:
                macs += out_channels * output_h * output_w
            flops = macs * 2  # Multiply-add counts as 2 operations
    
    elif layer_type == "Linear":
        # FLOPs = (in_features * out_features + out_features) * batch_size
        in_features = layer.in_features if hasattr(layer, 'in_features') else input_shape[-1]
        out_features = layer.out_features if hasattr(layer, 'out_features') else in_features
        batch_size = input_shape[0] if len(input_shape) > 1 else 1
        
        macs = in_features * out_features * batch_size
        if hasattr(layer, 'bias') and layer.bias is not None:
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
            kernel_size = layer.kernel_size if hasattr(layer, 'kernel_size') else (2, 2)
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            
            stride = layer.stride if hasattr(layer, 'stride') else kernel_size
            if isinstance(stride, int):
                stride = (stride, stride)
            
            padding = layer.padding if hasattr(layer, 'padding') else 0
            if isinstance(padding, int):
                padding = (padding, padding)
            
            output_h = (input_shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
            output_w = (input_shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
            
            total_elements = input_shape[0] * input_shape[1] * output_h * output_w
            flops = total_elements * kernel_size[0] * kernel_size[1]
    
    elif layer_type == "Upsample" or layer_type == "Interpolate":
        # FLOPs = output_size (simple copying/interpolation)
        if len(input_shape) >= 4:
            scale_factor = layer.scale_factor if hasattr(layer, 'scale_factor') else 2
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
        
        if len(children) == 0:
            # Leaf module - calculate FLOPs directly
            input_shape = tuple(current_x.shape) if isinstance(current_x, jt.Var) else (1,)
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
        else:
            # Container module - traverse children
            for name, child in children:
                full_name = f"{prefix}.{name}" if prefix else name
                current_x = traverse_and_calculate(child, current_x, prefix=full_name)
            return current_x
    
    # Set model to eval mode for profiling
    model_was_training = model.training if hasattr(model, 'training') else False
    model.eval()
    
    with jt.no_grad():
        _ = traverse_and_calculate(model, x)
    if hasattr(model, 'train'):
        model.train(model_was_training)

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
    # Handle inputs format (thop accepts inputs as list)
    if not isinstance(inputs, list):
        inputs = [inputs]
    
    # Ensure inputs are Jittor Vars
    inputs = [jt.array(inp) if not isinstance(inp, jt.Var) else inp for inp in inputs]
    
    # Calculate FLOPs and params
    flops, params = profile_model_recursive(model, inputs, verbose=verbose)
    
    return (flops, params)


def is_torch_model(model, inputs=None):
    """Check if the model is a PyTorch model.
    
    Args:
        model: Model/module to check
        inputs: Optional input tensor(s) or list of input tensors
    
    Returns:
        bool: True if the model is PyTorch, False otherwise
    """
    # Priority 1: Check input tensor type (most reliable)
    if inputs is not None:
        if not isinstance(inputs, list):
            inputs_list = [inputs]
        else:
            inputs_list = inputs
        
        if len(inputs_list) > 0:
            first_input = inputs_list[0]
            
            # Check if input is Jittor Var (not torch)
            if isinstance(first_input, jt.Var):
                return False
            
            # Check if input is PyTorch Tensor
            if TORCH_AVAILABLE:
                torch = _get_torch()
                if torch and isinstance(first_input, torch.Tensor):
                    return True
    
    # Priority 2: Check model type
    # Check if model is Jittor nn.Module (not torch)
    if isinstance(model, nn.Module):
        return False
    
    # Check if model is PyTorch nn.Module
    if TORCH_AVAILABLE:
        torch = _get_torch()
        if torch:
            torch_nn = torch.nn
            if isinstance(model, torch_nn.Module):
                return True
    
    # Priority 3: Check module name and attributes
    if hasattr(model, '__module__'):
        module_name = model.__module__
        if 'jittor' in module_name.lower():
            return False
        if TORCH_AVAILABLE and 'torch' in module_name.lower():
            return True
    
    # Check for framework-specific attributes
    if hasattr(model, 'execute'):
        return False  # Jittor has execute method
    if hasattr(model, 'forward') and not hasattr(model, 'execute'):
        return True  # PyTorch typically has forward but not execute
    
    return False


def _detect_framework(model, inputs):
    """Detect whether the model and inputs are PyTorch or Jittor.
    
    Args:
        model: Model/module to check
        inputs: Input tensor(s) or list of input tensors
    
    Returns:
        str: 'pytorch', 'jittor', or None if undetermined
    """
    # Normalize inputs to list
    if not isinstance(inputs, list):
        inputs_list = [inputs]
    else:
        inputs_list = inputs
    
    # Priority 1: Check input tensor type (most reliable)
    if len(inputs_list) > 0:
        first_input = inputs_list[0]
        
        # Check if input is Jittor Var
        if isinstance(first_input, jt.Var):
            return 'jittor'
        
        # Check if input is PyTorch Tensor
        if TORCH_AVAILABLE:
            torch = _get_torch()
            if torch and isinstance(first_input, torch.Tensor):
                return 'pytorch'
    
    # Priority 2: Check model type
    # Check if model is Jittor nn.Module
    if isinstance(model, nn.Module):
        return 'jittor'
    
    # Check if model is PyTorch nn.Module
    if TORCH_AVAILABLE:
        torch = _get_torch()
        if torch:
            torch_nn = torch.nn
            if isinstance(model, torch_nn.Module):
                return 'pytorch'
    
    # Priority 3: Check module name and attributes
    if hasattr(model, '__module__'):
        module_name = model.__module__
        if 'jittor' in module_name.lower():
            return 'jittor'
        if TORCH_AVAILABLE and 'torch' in module_name.lower():
            return 'pytorch'
    
    # Check for framework-specific attributes
    if hasattr(model, 'execute'):
        return 'jittor'
    if hasattr(model, 'forward') and not hasattr(model, 'execute'):
        return 'pytorch'
    
    return None


def profile(model, inputs, verbose=False, *args, **kwargs):
    """Unified profile function that automatically detects framework and uses appropriate tool.
    
    This function automatically detects whether the model and inputs are PyTorch or Jittor,
    and calls the appropriate profiling tool:
    - PyTorch: uses thop.profile (if available)
    - Jittor: uses native Jittor profile implementation
    
    Args:
        model: Model/module (PyTorch or Jittor)
        inputs: Input tensor(s) or list of input tensors
        verbose: Whether to print detailed information
        *args: Additional arguments (passed to thop.profile if using PyTorch)
        **kwargs: Additional keyword arguments (passed to thop.profile if using PyTorch)
    
    Returns:
        tuple: (flops, params) where:
            - flops: Total FLOPs count (raw number, not divided by 1e9)
            - params: Total number of parameters
    
    Example:
        ```python
        # Works with both PyTorch and Jittor
        import jittor as jt
        import jittor.nn as nn
        from nkyolo.utils.jittor_profile import profile
        
        # Jittor example
        model = nn.Conv2d(3, 64, 3, padding=1)
        x = jt.randn(1, 3, 224, 224)
        flops, params = profile(model, inputs=[x], verbose=True)
        
        # PyTorch example (if torch and thop are available)
        # import torch
        # import torch.nn as nn
        # model = nn.Conv2d(3, 64, 3, padding=1)
        # x = torch.randn(1, 3, 224, 224)
        # flops, params = profile(model, inputs=[x], verbose=True)
        ```
    """
    # Detect framework
    framework = _detect_framework(model, inputs)
    
    # Use appropriate profiling tool based on detected framework
    if framework == 'pytorch' and THOP_AVAILABLE:
        # Use original thop for PyTorch models
        thop_module = _get_thop()
        if thop_module:
            return thop_module.profile(model, inputs=inputs, verbose=verbose, *args, **kwargs)
    
    elif framework == 'jittor':
        # Use Jittor native profile for Jittor models
        return profile_jittor(model, inputs, verbose=verbose)
    
    # Framework not detected or thop not available
    # Try PyTorch/thop first if available, then fallback to Jittor
    if THOP_AVAILABLE and TORCH_AVAILABLE:
        thop_module = _get_thop()
        if thop_module:
            # Assume PyTorch and try thop
            return thop_module.profile(model, inputs=inputs, verbose=verbose, *args, **kwargs)
    
    # Default to Jittor method
    return profile_jittor(model, inputs, verbose=verbose)


def profile_layer_with_fallback(layer, inputs):
    """Profile a single layer with automatic framework detection and fallback.
    
    This function automatically detects if the layer is a PyTorch model:
    - If PyTorch: uses unified profile method
    - If not PyTorch (Jittor or other): uses fallback calculation method
    
    Args:
        layer: Layer/module to profile (PyTorch or Jittor)
        inputs: Input tensor(s) or list of input tensors
    
    Returns:
        float: FLOPs in GFLOPs (already divided by 1e9)
    """
    # Check if it's a PyTorch model
    if is_torch_model(layer, inputs):
        # Use unified profile for PyTorch models
        flops_result, _ = profile(layer, inputs=inputs, verbose=False)
        return flops_result / 1e9  # Convert to GFLOPs
    else:
        # Fallback to simple calculation for non-PyTorch models
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
        else:
            return 0.0


# For backward compatibility and thop-like usage
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
