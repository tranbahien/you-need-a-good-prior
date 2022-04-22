import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activation_fns import *
from ..layers.linear import Linear


def init_norm_layer(input_dim, norm_layer):
    if norm_layer == "batchnorm":
        return nn.BatchNorm1d(input_dim, eps=0, momentum=None,
                              affine=False, track_running_stats=False)
    elif norm_layer is None:
        return nn.Identity()


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation_fn,
                 scaled_variance=True, norm_layer=None,
                 task="regression"):
        """Initialization.

        Args:
            input_dim: int, the size of the input data.
            output_dim: int, the size of the output data.
            hidden_dims: list of int, the list containing the size of
                hidden layers.
            activation_fn: str, the name of activation function to be used
                in the network.
            norm_layer: str, the type of normaliztion layer applied after
                each layer
            task: string, the type of task, it should be either `regression`
                or `classification`.
        """
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.norm_layer = norm_layer
        self.task = task

        # Setup activation function
        options = {'cos': torch.cos, 'tanh': torch.tanh, 'relu': F.relu,
                   'softplus': F.softplus, 'rbf': rbf, 'linear': linear,
                   'sin': torch.sin, 'leaky_relu': F.leaky_relu, 
                   'swish': swish}
        if activation_fn in options:
            self.activation_fn = options[activation_fn]
        else:
            self.activation_fn = activation_fn

        self.layers = nn.ModuleList([Linear(
            input_dim, hidden_dims[0], scaled_variance=scaled_variance)])
        self.norm_layers = nn.ModuleList([init_norm_layer(
            hidden_dims[0], self.norm_layer)])
        for i in range(1, len(hidden_dims)):
            self.layers.add_module(
                "linear_{}".format(i), Linear(hidden_dims[i-1], hidden_dims[i],
                scaled_variance=scaled_variance))
            self.norm_layers.add_module(
                "norm_{}".format(i), init_norm_layer(hidden_dims[i],
                                                     self.norm_layer))
        self.output_layer = Linear(hidden_dims[-1], output_dim,
                                   scaled_variance=scaled_variance)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Linear):
                m.reset_parameters()

    def forward(self, X, log_softmax=False):
        """Performs forward pass given input data.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.
            log_softmax: bool, indicates whether or not return the log softmax
                values.

        Returns:
            torch.tensor, [batch_size, output_dim], the output data.
        """
        X = X.view(-1, self.input_dim)

        for linear_layer, norm_layer in zip(list(self.layers),
                                            list(self.norm_layers)):
            X = self.activation_fn(norm_layer(linear_layer(X)))

        X = self.output_layer(X)

        if (self.task == "classification") and log_softmax:
            X = F.log_softmax(X, dim=1)

        return X

    def predict(self, X):
        """Make predictions given input data.

        Args:
            x: torch tensor, shape [batch_size, input_dim]

        Returns:
            torch tensor, shape [batch_size, num_classes], the predicted
                probabilites for each class.
        """
        self.eval()
        if self.task == "classification":
            return torch.exp(self.forward(X, log_softmax=True))
        else:
            return self.forward(X, log_softmax=False)
