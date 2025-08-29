import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import log_loss

class TemperatureScaledModel(nn.Module):
    def __init__(self, model, temperature=1.0):
        super().__init__()
        self.model = model
        # Enable gradient computation for temperature parameter
        self.temperature = nn.Parameter(torch.tensor([temperature]), requires_grad=True)
    
    def forward(self, inputs):
        logits, *rest = self.model(inputs)
        return logits / self.temperature, *rest
    
    def temperature_scale(self, logits):
        """Apply temperature scaling to logits"""
        return logits / self.temperature

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE)
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine if sample is in bin m (between bin_lower and bin_upper)
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

# Usage example after calibration:
# temp_scaled_model = TemperatureScaledModel(trainer.model, temperature)
# torch.save(temp_scaled_model.state_dict(), 'temp_scaled_model.pth')
