import torch
import numpy as np
from typing import Union, List

def compute_pos_weight(labels: Union[List, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Compute positive class weight for binary classification
    
    Args:
        labels: Array-like of binary labels (0 or 1)
        
    Returns:
        torch.Tensor with single value - ratio of negative to positive samples
    """
    if isinstance(labels, (list, np.ndarray)):
        labels = torch.tensor(labels)
    
    # Count number of positive and negative samples
    n_negative = len(labels) - labels.sum()
    n_positive = labels.sum()
    
    # Compute weight as ratio of negative to positive samples
    pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    
    return torch.tensor([pos_weight])