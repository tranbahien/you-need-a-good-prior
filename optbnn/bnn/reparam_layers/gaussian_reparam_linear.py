import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianLinearReparameterization(nn.Module):
    def __init__(self, n_in, n_out, W_std=None,
                 b_std=None, scaled_variance=True):
        """Initialization.

        Args:
            n_in: int, the size of the input data.
            n_out: int, the size of the output.
            W_std: float, the initial value of
                the standard deviation of the weights.
            b_std: float, the initial value of
                the standard deviation of the biases.
        """
        super(GaussianLinearReparameterization, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.scaled_variance = scaled_variance

        if W_std is None:
            if self.scaled_variance:
                W_std = 1.
            else:
                W_std = 1. / math.sqrt(self.n_in)
        if b_std is None:
            b_std = 1.

        W_shape, b_shape = (1), (1)

        self.W_mu = 0.
        self.b_mu = 0.

        self.W_std = nn.Parameter(
            torch.ones(W_shape) * W_std,
            requires_grad=True)
        self.b_std = nn.Parameter(
            torch.ones(b_shape) * b_std, requires_grad=True)

    def forward(self, X):
        """Performs forward pass given input data.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.

        Returns:
            output: torch.tensor, [batch_size, output_dim], the output data.
        """
        W = self.W_mu + F.softplus(self.W_std) *\
            torch.randn((self.n_in, self.n_out), device=self.W_std.device)
        if self.scaled_variance:
            W = W / math.sqrt(self.n_in)
        b = self.b_mu + F.softplus(self.b_std) *\
            torch.randn((self.n_out), device=self.b_std.device)

        output = torch.mm(X, W) + b

        return output

    def sample_predict(self, X, n_samples):
        """Makes predictions using a set of sampled weights.

        Args:
            X: torch.tensor, [n_samples, batch_size, input_dim], the input
                data.
            n_samples: int, the number of weight samples used to make
                predictions.

        Returns:
            torch.tensor, [n_samples, batch_size, output_dim], the output data.
        """
        X = X.float()
        Ws = self.W_mu + F.softplus(self.W_std) *\
            torch.randn([n_samples, self.n_in, self.n_out],
                        device=self.W_std.device)

        if self.scaled_variance:
            Ws = Ws / math.sqrt(self.n_in)
        bs = self.b_mu + F.softplus(self.b_std) *\
            torch.randn([n_samples, 1, self.n_out],
                        device=self.b_std.device)

        return torch.matmul(X, Ws) + bs
