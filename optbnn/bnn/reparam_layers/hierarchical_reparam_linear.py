import math
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalLinearReparameterization(nn.Module):
    def __init__(self, n_in, n_out, W_shape=None, W_rate=None,
                 b_shape=None, b_rate=None, scaled_variance=True, eps=1e-8):
        super(HierarchicalLinearReparameterization, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.scaled_variance = scaled_variance
        self.eps = eps

        W_shape = 1. if W_shape is None else W_shape
        W_rate = 1. if W_rate is None else W_rate
        b_shape = 1. if b_shape is None else b_shape
        b_rate = 1. if b_rate is None else b_rate

        self.W_shape = nn.Parameter(torch.ones((1)) * W_shape, True)
        self.W_rate = nn.Parameter(torch.ones((1)) * W_rate, True)
        self.b_shape = nn.Parameter(torch.ones((1)) * b_shape, True)
        self.b_rate = nn.Parameter(torch.ones((1)) * b_rate, True)

    def _resample_std(self):
        # Non-negative constraints
        W_shape = F.softplus(self.W_shape)
        W_rate = F.softplus(self.W_rate)
        b_shape = F.softplus(self.b_shape)
        b_rate = F.softplus(self.b_rate)

        # Resample variances
        W_gamma_dist = dist.Gamma(W_shape, W_rate)
        b_gamma_dist = dist.Gamma(b_shape, b_rate)

        inv_W_var = W_gamma_dist.rsample()
        inv_b_var = b_gamma_dist.rsample()

        W_std = 1. / (torch.sqrt(inv_W_var) + self.eps)
        b_std = 1. / (torch.sqrt(inv_b_var) + self.eps)

        return W_std, b_std

    def forward(self, X):
        W_std, b_std = self._resample_std()

        # Resample parameters
        W = W_std * torch.randn((self.n_in, self.n_out),
                                device=self.W_shape.device)
        if self.scaled_variance:
            W = W / math.sqrt(self.n_in)
        b = b_std * torch.randn((self.n_out), device=self.b_shape.device)

        output = torch.mm(X, W) + b

        return output

    def sample_predict(self, X, n_samples):
        W_std, b_std = self._resample_std()

        X = X.float()

        Ws = W_std *\
            torch.randn([n_samples, self.n_in, self.n_out],
                        device=self.W_shape.device)

        if self.scaled_variance:
            Ws = Ws / math.sqrt(self.n_in)

        bs = b_std *\
            torch.randn([n_samples, 1, self.n_out],
                        device=self.b_shape.device)

        return torch.matmul(X, Ws) + bs