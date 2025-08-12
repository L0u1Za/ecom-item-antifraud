import torch
import torch.nn as nn

class GatingMechanism(nn.Module):
    def __init__(self, input_dim):
        super(GatingMechanism, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x1, x2):
        # Compute the gating weights
        gate = self.sigmoid(self.linear(x1 + x2))
        # Apply the gate to the inputs
        output = gate * x1 + (1 - gate) * x2
        return output