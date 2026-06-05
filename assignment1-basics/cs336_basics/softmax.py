import torch

def softmax(tensor:torch.Tensor, dim: int)-> torch.Tensor:
    """
    Compute the softmax of the input tensor along the specified dimension.
    
    Args:
        tensor: Input tensor of any shape.
        dim: The dimension along which to compute the softmax.
    
    Returns:
        A tensor of the same shape as the input, where the softmax has been applied along the specified dimension.
    """
    # Step 1: Subtract the maximum value along the specified dimension for numerical stability
    max_vals, _ = torch.max(tensor, dim=dim, keepdim=True)
    stabilized_tensor = tensor - max_vals
    
    # Step 2: Exponentiate the stabilized tensor
    exp_tensor = torch.exp(stabilized_tensor)
    
    # Step 3: Sum the exponentiated values along the specified dimension
    sum_exp = torch.sum(exp_tensor, dim=dim, keepdim=True)
    
    # Step 4: Divide the exponentiated values by the sum to get probabilities
    softmax_tensor = exp_tensor / sum_exp
    
    return softmax_tensor