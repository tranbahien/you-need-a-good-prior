import math
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalConv2dReparameterization(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, scaled_variance=True,
                 W_shape=None, W_rate=None, b_shape=None, b_rate=None,
                 eps=1e-8):
        super(HierarchicalConv2dReparameterization, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.scaled_variance = scaled_variance
        self.eps = eps

        W_shape = 1. if W_shape is None else W_shape
        W_rate = 1. if W_rate is None else W_rate
        b_shape = 1. if b_shape is None else b_shape
        b_rate = 1. if b_rate is None else b_rate

        self.W_shape = nn.Parameter(torch.ones((1)) * W_shape, True)
        self.W_rate = nn.Parameter(torch.ones((1)) * W_rate, True)

        if self.bias:
            self.b_shape = nn.Parameter(torch.ones((1)) * b_shape, True)
            self.b_rate = nn.Parameter(torch.ones((1)) * b_rate, True)

    def _resample_std(self):
        W_shape = F.softplus(self.W_shape)
        W_rate = F.softplus(self.W_rate)
        W_gamma_dist = dist.Gamma(W_shape, W_rate)
        inv_W_var = W_gamma_dist.rsample()
        W_std = 1. / (torch.sqrt(inv_W_var) + self.eps)

        if self.bias:
            b_shape = F.softplus(self.b_shape)
            b_rate = F.softplus(self.b_rate)
            b_gamma_dist = dist.Gamma(b_shape, b_rate)
            inv_b_var = b_gamma_dist.rsample()
            b_std = 1. / (torch.sqrt(inv_b_var) + self.eps)
        else:
            b_std = None

        return W_std, b_std

    def forward(self, X):
        W_std, b_std = self._resample_std()

        # Resample parameters
        W = W_std * torch.randn(
            (self.out_channels, self.in_channels, *self.kernel_size),
            device=self.W_shape.device)        

        if self.scaled_variance:
            W = W / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        if self.bias:
            b = b_std * \
                torch.randn((self.out_channels), device=self.b_shape.device)
        else:
            b = torch.zeros((self.out_channels), device=self.W_shape.device)

        return F.conv2d(X, W, b, self.stride, self.padding, self.dilation)

