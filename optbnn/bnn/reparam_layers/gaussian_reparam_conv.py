import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianConv2dReparameterization(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, W_std=None, b_std=None,
                 prior_per="layer", scaled_variance=True):
        super(GaussianConv2dReparameterization, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.scaled_variance = scaled_variance
        eps = 1e-6

        if W_std is None:
            if self.scaled_variance:
                W_std = 1.
            else:
                W_std = 1. / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        if b_std is None:
            b_std = 1.

        # Initialize the parameters
        if prior_per == "layer":
            W_shape, b_shape = (1), (1)
        elif prior_per == "parameter":
            W_shape = (self.out_channels, self.in_channels, *self.kernel_size)
            b_shape = (self.out_channels)

        self.W_mu = 0.
        self.b_mu = 0.

        self.W_std = nn.Parameter(
            torch.ones(W_shape) * W_std, requires_grad=True)
        if self.bias:
            self.b_std = nn.Parameter(
                torch.ones(b_shape) * b_std, requires_grad=True)
        else:
            self.register_buffer(
                'b_std', torch.ones(b_shape))

    def forward(self, X):
        W = self.W_mu + F.softplus(self.W_std) *\
            torch.randn((self.out_channels, self.in_channels,
                         *self.kernel_size), device=self.W_std.device)
        if self.scaled_variance:
            W = W / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        if self.bias:
            b = self.b_mu + F.softplus(self.b_std) *\
                torch.randn((self.out_channels), device=self.b_std.device)
        else:
            b = torch.zeros((self.out_channels), device=self.W_std.device)

        return F.conv2d(X, W, b, self.stride, self.padding, self.dilation)