import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Linear(nn.Module):
    def __init__(self, n_in, n_out, scaled_variance=True):
        """Initialization.

        Args:
            n_in: int, the size of the input data.
            n_out: int, the size of the output.
        """
        super(Linear, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.scaled_variance = scaled_variance

        # Initialize the parameters
        self.W = nn.Parameter(torch.zeros(self.n_in, self.n_out), True)
        self.b = nn.Parameter(torch.zeros(self.n_out), True)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.
        if not self.scaled_variance:
            std = std / math.sqrt(self.n_in)
        init.normal_(self.W, 0, std)
        init.constant_(self.b, 0)

    def forward(self, X):
        """Performs forward pass given input data.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.

        Returns:
            output: torch.tensor, [batch_size, output_dim], the output data.
        """
        W = self.W
        if self.scaled_variance:
            W = W / math.sqrt(self.n_in)
        b = self.b
        return torch.mm(X, W) + b
