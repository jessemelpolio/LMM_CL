import torch
import torch.nn as nn
from llava.utils import rank0_print
from transformers.integrations import is_deepspeed_zero3_enabled


def _create_buffer_copy_function():
    """Create a buffer copy function with DeepSpeed support if available."""
    # Try to import DeepSpeed's GatheredParameters
    if is_deepspeed_zero3_enabled():
        import deepspeed # Ensure import is local
        rank0_print("DeepSpeed ZeRO3 enabled, using GatheredParameters for buffer copying")
        
        # Return a function that uses GatheredParameters for buffers
        def copy_buffer_with_deepspeed(source_buffer, target_buffer):
            with deepspeed.zero.GatheredParameters([source_buffer, target_buffer], modifier_rank=0):
                # Only perform operations on rank 0 after gathering
                if torch.distributed.get_rank() == 0:
                    if source_buffer.shape != target_buffer.shape:
                        rank0_print(f"[Copy Buffer ERROR] Shape mismatch: source {source_buffer.shape} vs target {target_buffer.shape}")
                        raise ValueError(f"Buffer shape mismatch: source {source_buffer.shape} vs target {target_buffer.shape}")
                        
                    target_device = target_buffer.device # Get the device of the gathered target buffer
                    target_dtype = target_buffer.dtype # Get the dtype of the gathered target buffer
                    source_device = source_buffer.device
                    source_dtype = source_buffer.dtype

                    # Move source to target device if needed
                    if source_device != target_device:
                        # rank0_print(f"  [Copy Buffer] Moving source from {source_device} to {target_device}")
                        source_buffer_copy = source_buffer.to(device=target_device)
                    else:
                        source_buffer_copy = source_buffer
                    
                    # Handle dtype conversion - we want to convert the target to source dtype
                    if source_dtype != target_dtype:
                        # rank0_print(f"  [Copy Buffer] Converting target dtype from {target_dtype} to {source_dtype}")
                        # Convert target data to source dtype before copying
                        # This is because copy_() preserves target dtype, not source dtype
                        target_buffer.data = target_buffer.data.to(dtype=source_dtype)
                    
                    # Now copy the values
                    target_buffer.copy_(source_buffer_copy)

                    # # verify copy
                    # if torch.allclose(target_buffer, source_buffer_copy):
                    #     rank0_print(f"  [Copy Buffer] Copy successful")
                    # else:
                    #     rank0_print(f"  [Copy Buffer] Copy failed")
                    #     return False
                    
                    # rank0_print(f"Now target buffer dtype is {target_buffer.data.dtype}")
                    # rank0_print(f"  [Copy Buffer] Copy successful")
        return copy_buffer_with_deepspeed
        
    else:
        rank0_print("DeepSpeed ZeRO3 not enabled, using direct buffer copying")
        
        # Return a simple copy function for buffers
        def copy_buffer_direct(source_buffer, target_buffer):
            if source_buffer.shape != target_buffer.shape:
                rank0_print(f"[Copy Buffer ERROR] Shape mismatch: source {source_buffer.shape} vs target {target_buffer.shape}")
                raise ValueError(f"Buffer shape mismatch: source {source_buffer.shape} vs target {target_buffer.shape}")
                
            target_device = target_buffer.device
            target_dtype = target_buffer.dtype
            source_device = source_buffer.device
            source_dtype = source_buffer.dtype
            
            # Move source to target device if needed
            if source_device != target_device:
                # rank0_print(f"  [Copy Buffer Direct] Moving source from {source_device} to {target_device}")
                source_buffer_copy = source_buffer.to(device=target_device)
            else:
                source_buffer_copy = source_buffer
            
            # Handle dtype conversion - we want to convert the target to source dtype
            if source_dtype != target_dtype:
                # rank0_print(f"  [Copy Buffer Direct] Converting target dtype from {target_dtype} to {source_dtype}")
                # Convert target data to source dtype before copying
                # This is because copy_() preserves target dtype, not source dtype
                target_buffer.data = target_buffer.data.to(dtype=source_dtype)
            
            # Now copy the values
            target_buffer.copy_(source_buffer_copy)

            # verify copy
            # if torch.allclose(target_buffer, source_buffer_copy):
            #     rank0_print(f"  [Copy Buffer Direct] Copy successful")
            # else:
            #     rank0_print(f"  [Copy Buffer Direct] Copy failed")
            #     return False
            
            return True
        return copy_buffer_direct


def _create_parameter_copy_function():
    """Create a parameter copy function with DeepSpeed support if available."""
    # Try to import DeepSpeed's GatheredParameters
    if is_deepspeed_zero3_enabled():
        import deepspeed # Ensure import is local
        rank0_print("DeepSpeed ZeRO3 enabled, using GatheredParameters for safe parameter copying")
        
        # Return a function that uses GatheredParameters
        def copy_with_deepspeed(source_param, target_param):
            with deepspeed.zero.GatheredParameters([source_param, target_param], modifier_rank=0):
                gathered_source_data = getattr(source_param, 'data', None)
                gathered_target_data = getattr(target_param, 'data', None)
                
                # Only perform operations on rank 0 after gathering
                if torch.distributed.get_rank() == 0:
                    if (gathered_source_data is None or gathered_target_data is None):
                        rank0_print(f"[Copy Param ERROR] Gathered parameter data is None")
                        return False
                        
                    if gathered_source_data.shape != gathered_target_data.shape:
                        rank0_print(f"[Copy Param ERROR] Shape mismatch: source {gathered_source_data.shape} vs target {gathered_target_data.shape}")
                        raise ValueError(f"Parameter shape mismatch: source {gathered_source_data.shape} vs target {gathered_target_data.shape}")
                        
                    target_device = gathered_target_data.device 
                    target_dtype = gathered_target_data.dtype
                    source_device = gathered_source_data.device
                    source_dtype = gathered_source_data.dtype

                    # Move source to target device if needed
                    if source_device != target_device:
                        # rank0_print(f"  [Copy Param] Moving source from {source_device} to {target_device}")
                        source_data_copy = gathered_source_data.to(device=target_device)
                    else:
                        source_data_copy = gathered_source_data
                    
                    # Handle dtype conversion - we want to convert the target to source dtype
                    if source_dtype != target_dtype:
                        # rank0_print(f"  [Copy Param] Converting target dtype from {target_dtype} to {source_dtype}")
                        # Convert target data to source dtype before copying
                        # This is because copy_() preserves target dtype, not source dtype
                        gathered_target_data.data = gathered_target_data.data.to(dtype=source_dtype) 
                    
                    # Now copy the values
                    gathered_target_data.copy_(source_data_copy)
                    
                    return True
                
                # All ranks return success (actual copying only happens on rank 0)
                return True
        return copy_with_deepspeed
        
    else:
        rank0_print("DeepSpeed ZeRO3 not enabled, using direct parameter copying")
        
        # Return a simple copy function
        def copy_direct(source_param, target_param):
            if (source_param.data is None or target_param.data is None):
                rank0_print(f"[Copy Param ERROR] Source or target parameter data is None")
                return False
                
            if source_param.data.shape != target_param.data.shape:
                rank0_print(f"[Copy Param ERROR] Shape mismatch: source {source_param.data.shape} vs target {target_param.data.shape}")
                return False
            
            target_device = target_param.data.device
            target_dtype = target_param.data.dtype
            source_device = source_param.data.device
            source_dtype = source_param.data.dtype
            
            # Move source to target device if needed
            if source_device != target_device:
                # rank0_print(f"  [Copy Param Direct] Moving source from {source_device} to {target_device}")
                source_param_copy = source_param.data.to(device=target_device)
            else:
                source_param_copy = source_param.data
            
            # Handle dtype conversion - we want to convert the target to source dtype
            if source_dtype != target_dtype:
                # rank0_print(f"  [Copy Param Direct] Converting target dtype from {target_dtype} to {source_dtype}")
                # Convert target data to source dtype before copying
                # This is because copy_() preserves target dtype, not source dtype
                target_param.data = target_param.data.to(dtype=source_dtype)
            
            # Now copy the values
            target_param.data.copy_(source_param_copy)

            # # verify copy
            # if torch.allclose(target_param.data, source_param_copy):
            #     rank0_print(f"  [Copy Param Direct] Copy successful")
            # else:
            #     rank0_print(f"  [Copy Param Direct] Copy failed")
            #     return False
            
            return True
            
        return copy_direct


def copy_module_params(source_module, target_module, prefix="", copy_fn=_create_parameter_copy_function()):
    """
    Recursively copy parameters from source module to target module,
    handling both direct parameters and nested modules.
    
    Args:
        source_module: Source module or parameter
        target_module: Target module or parameter
        prefix: Parameter name prefix for logging (used in recursion)
        copy_fn: The function to use for actual parameter copying (handles DeepSpeed)
    
    Returns:
        bool: True if copy was successful, False otherwise
    """
    if source_module is None or target_module is None:
        if prefix:
            rank0_print(f"Source or target module is None at prefix {prefix}")
        return False

    if copy_fn is None:
        raise ValueError("copy_fn must be provided to copy_module_params")

    # Base case: Direct parameter copy
    if isinstance(source_module, nn.Parameter) and isinstance(target_module, nn.Parameter):
        copy_success = copy_fn(source_module, target_module)
        if not copy_success and prefix:
            rank0_print(f"Failed to copy parameter: {prefix}")
        return copy_success

    # Recursive case: Both are modules
    if isinstance(source_module, nn.Module) and isinstance(target_module, nn.Module):
        overall_success = True # Track success for this module level

        # 1. Copy direct parameters (_parameters dictionary)
        source_params = dict(source_module.named_parameters(recurse=False)) # Use named_parameters(recurse=False)
        target_params = dict(target_module.named_parameters(recurse=False))

        for name, param in source_params.items():
            current_prefix = f"{prefix}.{name}" if prefix else name
            if name in target_params:
                target_param = target_params[name]
                # Both should be nn.Parameter here, no need for None check?
                # Add safety check just in case
                if not isinstance(param, nn.Parameter) or not isinstance(target_param, nn.Parameter):
                     rank0_print(f"Type mismatch for direct parameter {current_prefix}: Source {type(param)}, Target {type(target_param)}")
                     overall_success = False
                     continue # Skip if types are wrong

                param_success = copy_fn(param, target_param)
                if not param_success:
                    rank0_print(f"Failed copy for direct parameter: {current_prefix}")
                overall_success = overall_success and param_success
            else:
                rank0_print(f"Parameter {name} not found in target module {prefix}. Skipping.")
                overall_success = False

        # 2. Recursively copy child modules (_modules dictionary)
        source_children = dict(source_module.named_children()) # Use named_children()
        target_children = dict(target_module.named_children())

        for name, child_module in source_children.items():
            current_prefix = f"{prefix}.{name}" if prefix else name
            if name in target_children:
                target_child_module = target_children[name]
                # Both should be nn.Module here
                if not isinstance(child_module, nn.Module) or not isinstance(target_child_module, nn.Module):
                    rank0_print(f"Type mismatch for child module {current_prefix}: Source {type(child_module)}, Target {type(target_child_module)}")
                    overall_success = False
                    continue # Skip if types are wrong

                child_success = copy_module_params(
                    child_module,
                    target_child_module,
                    current_prefix,
                    copy_fn
                )
                overall_success = overall_success and child_success
            else:
                rank0_print(f"Module {name} not found in target module {prefix}. Skipping.")
                overall_success = False

        return overall_success

    # Handle type mismatch (e.g., copying Module to Parameter)
    else:
        if prefix:
            rank0_print(f"Type mismatch at top level {prefix}: Source type {type(source_module)}, Target type {type(target_module)}")
        return False
    

# Helper function to list all buffers recursively
def list_buffers_recursively(module, prefix=""):
    """Recursively list all buffers in a module with their path.
    Returns a list of (path, buffer) tuples.
    """
    buffers = []
    
    # Add direct buffers
    for name, buffer in module._buffers.items():
        if buffer is not None:
            path = f"{prefix}.{name}" if prefix else name
            buffers.append((path, buffer))
    
    # Add buffers from child modules
    for name, child in module._modules.items():
        if child is not None:
            child_prefix = f"{prefix}.{name}" if prefix else name
            buffers.extend(list_buffers_recursively(child, child_prefix))
    
    return buffers

def list_params_recursively(module, prefix=""):
    """Recursively list all parameters in a module with their path.
    Returns a list of (path, parameter) tuples.
    """
    params = []
    
    # Add direct parameters
    for name, param in module.named_parameters(recurse=False):
        if param is not None:
            path = f"{prefix}.{name}" if prefix else name
            params.append((path, param))
    
    # Add parameters from child modules
    for name, child in module.named_children():
        if child is not None:
            child_prefix = f"{prefix}.{name}" if prefix else name
            params.extend(list_params_recursively(child, child_prefix))
    
    return params

# Helper function to get buffer by path
def get_buffer_by_path(module, path):
    """Utility to get a buffer by its path string.
    
    Args:
        module: Module to get buffer from
        path: String path to buffer, e.g., "self_attn.q_proj.weight"
        
    Returns:
        Buffer if found, None otherwise
    """
    parts = path.split('.')
    current = module
    
    # Navigate through the module tree until the final component
    for i, part in enumerate(parts[:-1]):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
        if current is None:
            return None
    
    # Get the final buffer
    final_part = parts[-1]
    if not hasattr(current, final_part):
        return None
    
    final_obj = getattr(current, final_part)
    if final_obj is None or not isinstance(final_obj, torch.Tensor):
        return None  # Not a tensor buffer
        
    return final_obj

# Helper function to get parameter by path
def get_param_by_path(module, path):
    """Utility to get a parameter by its path string.
    
    Args:
        module: Module to get parameter from
        path: String path to parameter, e.g., "self_attn.q_proj.weight"
        
    Returns:
        Parameter if found, None otherwise
    """
    parts = path.split('.')
    current = module
    
    # Navigate through the module tree until the final component
    for i, part in enumerate(parts[:-1]):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
        if current is None:
            return None
    
    # Get the final parameter
    final_part = parts[-1]
    if not hasattr(current, final_part):
        return None
    
    final_obj = getattr(current, final_part)
    if final_obj is None or not isinstance(final_obj, nn.Parameter):
        return None  # Not a parameter
        
    return final_obj
