import torch

def debug_shapes(prefix, **tensors):
    """
    Print shapes of multiple tensors with a prefix for easy identification
    
    Args:
        prefix (str): Prefix for the debug statement
        **tensors: Dictionary of tensors to print shapes for
    """
    shapes = {name: tensor.shape if hasattr(tensor, 'shape') else None 
              for name, tensor in tensors.items()}
    print(f"{prefix} - Shapes: {shapes}")

def inspect_tensor(name, tensor, full_stats=False):
    """
    Print detailed information about a tensor
    
    Args:
        name (str): Name of the tensor
        tensor (torch.Tensor): The tensor to inspect
        full_stats (bool): Whether to include min/max/mean/std statistics
    """
    print(f"\n--- {name} Inspection ---")
    print(f"Shape: {tensor.shape}")
    print(f"Device: {tensor.device}")
    print(f"Dtype: {tensor.dtype}")
    
    if full_stats:
        print(f"Min value: {tensor.min().item()}")
        print(f"Max value: {tensor.max().item()}")
        print(f"Mean value: {tensor.mean().item()}")
        print(f"Std deviation: {tensor.std().item()}")
        
        # Check for NaN or infinite values
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        print(f"Contains NaN: {has_nan}")
        print(f"Contains Inf: {has_inf}")
    
    # Print small sample of data
    flat_tensor = tensor.view(-1)
    sample_size = min(5, flat_tensor.numel())
    if sample_size > 0:
        print(f"First {sample_size} values: {flat_tensor[:sample_size].tolist()}")
    print("------------------------\n")

# Example usage in training loop:
# from debug_utils import debug_shapes, inspect_tensor
# 
# debug_shapes("Before discriminator", fake_adj=fake_adj, target_m=target_m)
# inspect_tensor("Generated adjacency matrix", fake_adj, full_stats=True)