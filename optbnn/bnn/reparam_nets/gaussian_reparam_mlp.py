import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activation_fns import *
from ..reparam_layers.gaussian_reparam_linear import GaussianLinearReparameterization


def init_norm_layer(input_dim, norm_layer):
    if norm_layer == "batchnorm":
        return nn.BatchNorm1d(input_dim, eps=0, momentum=None,
                              affine=False, track_running_stats=False)
    elif norm_layer is None:
        return nn.Identity()


class GaussianMLPReparameterization(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation_fn,
                 W_std=None, b_std=None, scaled_variance=True, norm_layer=None):
        """Initialization.

        Args:
            input_dim: int, the size of the input data.
            output_dim: int, the size of the output data.
            hidden_dims: list of int, the list containing the size of
                hidden layers.
            activation_fn: str, the name of activation function to be used
                in the network.
            W_std: float, the initial value of the logarithm of
                the standard deviation of the weights.
            b_std: float, the initial value of the logarithm of
                the standard deviation of the biases.
        """
        super(GaussianMLPReparameterization, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.norm_layer = norm_layer

        # Setup activation function
        options = {'cos': torch.cos, 'tanh': torch.tanh, 'relu': F.relu,
                   'softplus': F.softplus, 'rbf': rbf, 'linear': linear,
                   'sin': sin, 'leaky_relu': F.leaky_relu,
                   'swish': swish}
        if activation_fn in options:
            self.activation_fn = options[activation_fn]
        else:
            self.activation_fn = activation_fn

        if b_std is None:
            b_std = W_std

        # Initialize layers
        self.layers = nn.ModuleList([GaussianLinearReparameterization(
            input_dim, hidden_dims[0], W_std, b_std,
            scaled_variance=scaled_variance)])

        self.norm_layers = nn.ModuleList([init_norm_layer(
            hidden_dims[0], self.norm_layer)])

        for i in range(1, len(hidden_dims)):
            self.layers.add_module(
                "linear_{}".format(i), GaussianLinearReparameterization(
                    hidden_dims[i-1], hidden_dims[i], W_std, b_std,
                    scaled_variance=scaled_variance))
            self.norm_layers.add_module(
                "norm_{}".format(i), init_norm_layer(hidden_dims[i],
                                                     self.norm_layer))

        self.output_layer = GaussianLinearReparameterization(
            hidden_dims[-1], output_dim, W_std, b_std,
            scaled_variance=scaled_variance)

    def forward(self, X):
        """Performs forward pass given input data.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.
            sample: boolean, whether or not perform forward pass using
                sampled weights.

        Returns:
            torch.tensor, [batch_size, output_dim], the output data.
        """
        X = X.view(-1, self.input_dim)

        for linear_layer, norm_layer in zip(list(self.layers),
                                            list(self.norm_layers)):
            X = self.activation_fn(norm_layer(linear_layer(X)))

        X = self.output_layer(X)

        return X

    def sample_functions(self, X, n_samples):
        """Performs predictions using `n_samples` set of weights.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.
            n_samples: int, the number of weight samples used to make
                predictions.

        Returns:
            torch.tensor, [batch_size, n_samples, output_dim], the output
            data.
        """
        X = X.view(-1, self.input_dim)
        X = torch.unsqueeze(X, 0).repeat([n_samples, 1, 1])
        for linear_layer, norm_layer in zip(list(self.layers),
                                            list(self.norm_layers)):
            if self.norm_layer is None:
                X = self.activation_fn(linear_layer.sample_predict(X, n_samples))
            else:
                X = linear_layer.sample_predict(X, n_samples)
                out = torch.zeros_like(X, device=X.device, dtype=X.dtype)
                for i in range(n_samples):
                    out[i, :, :] = norm_layer(X[i, :, :])
                X = self.activation_fn(out)

        X = self.output_layer.sample_predict(X, n_samples)
        X = torch.transpose(X, 0, 1)

        return X
