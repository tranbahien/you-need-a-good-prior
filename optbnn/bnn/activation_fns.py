"""Define activation functions for neural network"""

__all__ = ['rbf', 'linear', 'sin', 'cos', 'swish']

import torch
import torch.nn.functional as F


# RBF function
rbf = lambda x: torch.exp(-x**2)

# Linear function
linear = lambda x: x

# Sin function
sin = lambda x: torch.sin(x)

# Cos function
cos = lambda x: torch.cos(x)

# Swiss function
swish = lambda x: x * torch.sigmoid(x)

softplus = lambda x: F.softplus(x)
