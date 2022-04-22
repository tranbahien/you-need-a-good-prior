"""Defines some utilities for running experiments"""

import torch
import numpy as np


def median_distance_local(x, default_value=1.0, eps=1e-6):
    """Get the median of distances between x.

    Args:
        x: numpy array [n_data, n_dim], the input data.
        default_value: float, the default value of distance in the case
            the distance is too small (< eps).
    """
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x, 0)
    dis_a = np.abs(x_col - x_row) # [n, n, d]
    dis_a = np.reshape(dis_a, [-1, dis_a.shape[-1]])
    dis = np.median(dis_a, 0) * (x.shape[1] ** 0.5)
    if default_value is not None:
        dis[np.abs(dis) < eps] = default_value

    return dis


def get_input_range(X_train, X_test, ratio=0.0):
    """Get the range of each coordinate of the input data."""
    x_min = np.minimum(X_train.min(axis=0), X_test.min(axis=0))
    x_max = np.maximum(X_train.max(axis=0), X_test.max(axis=0))
    d = x_max - x_min
    x_min = x_min - d * ratio
    x_max = x_max + d * ratio

    return x_min, x_max


def optimize_gp_model(gp, lr, num_iters, logger=None):
    """Optimize kernel's hyperparametr of GP."""
    gp_optimizer = torch.optim.LBFGS(gp.parameters(), lr=lr, max_iter=num_iters)

    def _eval_model():
        obj = gp()
        gp_optimizer.zero_grad()
        obj.backward()
        return obj

    for i in range(num_iters):
        obj = gp()
        gp_optimizer.zero_grad()
        obj.backward()
        gp_optimizer.step(_eval_model)
        info_txt = ">>> Iteration # {:3d}: NLL: {:.4f}".format(i, obj.item())
        if logger is not None:
            logger.info(info_txt)
        else:
            print(info_txt)
    return gp
