"""Defines likelihood of some distributions."""

import torch
import torch.nn as nn


class LikelihoodModule(nn.Module):
    def forward(self, fx, y):
        return -self.loglik(fx, y)

    def loglik(self, fx, y):
        raise NotImplementedError


class LikGaussian(LikelihoodModule):
    def __init__(self, var):
        super(LikGaussian, self).__init__()
        self.loss = torch.nn.MSELoss(reduction='sum')
        self.var = var

    def loglik(self, fx, y):
        return - 0.5 / self.var * self.loss(fx, y)


class LikLaplace(LikelihoodModule):
    def __init__(self, scale):
        super(LikLaplace, self).__init__()
        self.loss = torch.nn.L1Loss(reduction='sum')
        self.scale = scale

    def loglik(self, fx, y):
        return - 1 / self.scale * self.loss(fx, y)


class LikBernoulli(LikelihoodModule):
    def __init__(self):
        super(LikBernoulli, self).__init__()
        self.loss = torch.nn.BCELoss(reduction='sum')

    def loglik(self, fx, y):
        return -self.loss(fx, y)


class LikCategorical(LikelihoodModule):
    def __init__(self):
        super(LikCategorical, self).__init__()
        self.loss = torch.nn.NLLLoss(reduction='sum')

    def loglik(self, fx, y):
        return -self.loss(fx, y)
