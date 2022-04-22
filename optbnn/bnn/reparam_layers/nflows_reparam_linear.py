import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nflows import NormalizingFlow


class NormFlowsLinearReparameterization(nn.Module):
    """A wrapper of Re-parameterized linear layer with normalising flows."""
    def __init__(self, n_in, n_out, n_transformations=4, scaled_variance=True, bias=True):
        """Initialization.

        Args:
            n_in: int, the size of the input data.
            n_out: int, the size of the output.
            n_transformations: Number of nflow transformations
        """
        super(NormFlowsLinearReparameterization, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.scaled_variance = scaled_variance
        self.use_bias = bias
        if bias:
            '''
            The 'bias' will be as an extra dimension in the NormalizingFlow
            Note: self.n_in will not change
            '''
            n_in += 1
        self.W_prior = NormalizingFlow(n_in, n_out, n_transformations=n_transformations)

    def forward(self, X):
        """Performs forward pass given input data.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.

        Returns:
            output: torch.tensor, [batch_size, output_dim], the output data.
        """
        return self.sample_predict(X, 1)

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
        W_samples = self.W_prior.sample(n_samples) ##size: [n_samples, n_in, n_out]
        if self.use_bias:
            bs = W_samples[:, :1, :]
            Ws = W_samples[:, 1:, :]
            if self.scaled_variance:
                Ws = Ws / np.sqrt(self.n_in)
            return torch.matmul(X, Ws) + bs
        else:
            if self.scaled_variance:
                W_samples = W_samples / np.sqrt(self.n_in)
            return torch.matmul(X, W_samples)

