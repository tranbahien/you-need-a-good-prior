"""Defines prior modules."""

import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class PriorModule(nn.Module):
    """Generic class of Prior module"""
    def __init__(self):
        super(PriorModule, self).__init__()
        self.hyperprior = False

    def forward(self, net):
        """Compute the negative log likelihood.

        Args:
            net: nn.Module, the input network needs to be evaluated.
        """
        return -self.logp(net)

    def initialize(self, net):
        """Initialize neural network's parameters according to the prior.

        Args:
            net: nn.Module, the input network needs to be initialzied.
        """
        for name, param in net.named_parameters():
            if param.requires_grad:
                value = self.sample(name, param)
                if value is not None:
                    param.data.copy_(value)

    def logp(self, net):
        """Compute the log likelihood

        Args:
            net: nn.Module, the input network needs to be evaluated.
        """
        raise NotImplementedError

    def sample(self, name, param):
        """Sample parameters from prior.

        Args:
            name: str, the name of the parameter.
            param: torch.Parameter, the parameter need to be sampled.
        """
        raise NotImplementedError


class FixedHierarchicalPrior(PriorModule):
    """Class of Gaussian prior. We further place Inv-Gamma hyper-prior
        on variance of Gaussian prior for each layer."""
    def __init__(self, net, shape=1.0, rate=1.0, eps=1e-8):
        """Initialization."""
        super(FixedHierarchicalPrior, self).__init__()

        self.hyperprior = True
        self.shape = torch.Tensor([shape])
        self.rate = torch.Tensor([rate])
        self.eps = eps

        self.params = {}
        self._initialize(net)

    def _sample_std(self, shape, rate):
        with torch.no_grad():
            gamma_dist = dist.Gamma(shape, rate)
            inv_var = gamma_dist.rsample()
            std = 1. / (torch.sqrt(inv_var) + self.eps)

            return std

    def resample(self, net):
        """Sample std of Gaussian prior for each layer by using
            Gibb sampler."""
        for name, param in net.named_parameters():
            if ('.W' in name) or ('.b' in name):
                sumcnt = param.data.nelement()
                sumsqr = (param.data ** 2).sum().item()

                shape_ = self.shape + 0.5 * sumcnt
                rate_ = self.rate + 0.5 * sumsqr
                std = self._sample_std(shape_, rate_)

                if '.W' in name:
                    self.params[name.replace('.W', '.W_std')] = std
                if '.b' in name:
                    self.params[name.replace('.b', '.b_std')] = std

    def _initialize(self, net):
        for name, param in net.named_parameters():
            if '.W' in name:
                name_mu = name.replace('.W', '.W_mu')
                name_std = name.replace('.W', '.W_std')
                self.params[name_mu] = 0.0
                self.params[name_std] = self._sample_std(self.shape, self.rate)
            elif '.b' in name:
                name_mu = name.replace('.b', '.b_mu')
                name_std = name.replace('.b', '.b_std')
                self.params[name_mu] = 0.0
                self.params[name_std] = self._sample_std(self.shape, self.rate)

    def _get_params_by_name(self, name):
        """Get the paramters of prior by name."""

        if not (('.W' in name) or ('.b' in name)):
            return None, None
        mu, std = None, None

        if '.W' in name:
            name_mu = name.replace('.W', '.W_mu')
            name_std = name.replace('.W', '.W_std')

            if name_mu in self.params.keys():
                mu = self.params[name_mu]
            if name_std in self.params.keys():
                std = self.params[name_std]
        elif '.b' in name:
            name_mu = name.replace('.b', '.b_mu')
            name_std = name.replace('.b', '.b_std')

            if name_mu in self.params.keys():
                mu = self.params[name_mu]
            if name_std in self.params.keys():
                std = self.params[name_std]

        return mu, std

    def logp(self, net):
        """Compute the log likelihood

        Args:
            net: nn.Module, the input network needs to be evaluated.
        """
        res = 0.
        for name, param in net.named_parameters():
            mu, std = self._get_params_by_name(name)
            if (mu is None) and (std is None):
                continue
            var = std.to(param.device) ** 2
            res -= torch.sum(((param - mu) ** 2) / (2 * var))
        return res


class FixedGaussianPrior(PriorModule):
    """Class of Standard Gaussian Prior."""
    def __init__(self, mu=0.0, std=1.0, device="cpu"):
        """Initialization."""
        super(FixedGaussianPrior, self).__init__()

        self.mu = mu
        self.std = std

    def sample(self, name, param):
        """Sample parameters from prior.

        Args:
            name: str, the name of the parameter.
            param: torch.Parameter, the parameter need to be sampled.
        """
        mu, std = self._get_params_by_name(name)

        if (mu is None) and (std is None):
            return None

        return (mu + std * torch.randn_like(param)).to(param.device)

    def _get_params_by_name(self, name):
        """Get the paramters of prior by name."""

        if not (('.W' in name) or ('.b' in name)):
            return None, None
        else:
            return self.mu, self.std

    def logp(self, net):
        """Compute the log likelihood

        Args:
            net: nn.Module, the input network needs to be evaluated.
        """
        res = 0.
        for name, param in net.named_parameters():
            mu, std = self._get_params_by_name(name)
            if (mu is None) and (std is None):
                continue
            var = std ** 2
            res -= torch.sum(((param - mu) ** 2) / (2 * var))
        return res


class OptimHierarchicalPrior(PriorModule):
    def __init__(self, net, saved_path, device="cpu", eps=1e-8):
        super(OptimHierarchicalPrior, self).__init__()

        self.hyperprior = True
        self.params = {}
        self.device = device
        self.eps = eps

        data = torch.load(saved_path, map_location=torch.device(self.device))
        for name, param in data.items():
            self.params[name] = param.to(self.device)

        self._initialize(net)

    def to(self, device):
        for name in self.params.keys():
            self.params[name] = self.params[name].to(device)
        return self

    def _sample_std(self, shape, rate):
        with torch.no_grad():
            shape = F.softplus(shape)
            rate = F.softplus(rate)

            gamma_dist = dist.Gamma(shape, rate)
            inv_var = gamma_dist.rsample()
            std = 1. / (torch.sqrt(inv_var) + self.eps)

            return std

    def resample(self, net):
        for name, param in net.named_parameters():
            if ('.W' in name) or ('.b' in name):
                sumcnt = param.data.nelement()
                sumsqr = (param.data ** 2).sum().item()

                if '.W' in name:
                    shape = self.params[name.replace('.W', '.W_shape')]
                    rate = self.params[name.replace('.W', '.W_rate')]
                if '.b' in name:
                    shape = self.params[name.replace('.b', '.b_shape')]
                    rate = self.params[name.replace('.b', '.b_rate')]

                shape_ = shape + 0.5 * sumcnt
                rate_ = rate + 0.5 * sumsqr
                std = self._sample_std(shape_, rate_)

                if '.W' in name:
                    self.params[name.replace('.W', '.W_std')] = std
                if '.b' in name:
                    self.params[name.replace('.b', '.b_std')] = std

    def _initialize(self, net):
        for name, param in net.named_parameters():
            if '.W' in name:
                shape = self.params[name.replace('.W', '.W_shape')]
                rate = self.params[name.replace('.W', '.W_rate')]
                name_mu = name.replace('.W', '.W_mu')
                name_std = name.replace('.W', '.W_std')
                self.params[name_mu] = torch.tensor([0.0])
                self.params[name_std] = self._sample_std(shape, rate)
            elif '.b' in name:
                shape = self.params[name.replace('.b', '.b_shape')]
                rate = self.params[name.replace('.b', '.b_rate')]
                name_mu = name.replace('.b', '.b_mu')
                name_std = name.replace('.b', '.b_std')
                self.params[name_mu] = torch.tensor([0.0])
                self.params[name_std] = self._sample_std(shape, rate)

    def _get_params_by_name(self, name):
        if not (('.W' in name) or ('.b' in name)):
            return None, None
        mu, std = None, None

        if '.W' in name:
            name_mu = name.replace('.W', '.W_mu')
            name_std = name.replace('.W', '.W_std')

            if name_mu in self.params.keys():
                mu = self.params[name_mu]
            if name_std in self.params.keys():
                std = self.params[name_std]
        elif '.b' in name:
            name_mu = name.replace('.b', '.b_mu')
            name_std = name.replace('.b', '.b_std')

            if name_mu in self.params.keys():
                mu = self.params[name_mu]
            if name_std in self.params.keys():
                std = self.params[name_std]

        return mu, std

    def logp(self, net):
        res = 0.
        for name, param in net.named_parameters():
            mu, std = self._get_params_by_name(name)
            if (mu is None) and (std is None):
                continue
            var = std ** 2
            res -= torch.sum(((param - mu) ** 2) / (2 * var))

        return res


class OptimNormFlowPrior(PriorModule):
    def __init__(self, model, saved_path, device='cpu'):
        super(OptimNormFlowPrior, self).__init__()
        self.params = {}
        self.device = device
        self.model = model # type: NormFlowsLinearReparameterization
        ### Fix the model
        self.model.requires_grad_(False)
        
        if saved_path is not None:
            data = torch.load(saved_path, map_location=torch.device(self.device))
            self.model.load_state_dict(data)

    def logp(self, net: nn.Module):
        """Note: this is highly customized for the MLP case"""
        ## Get all hidden layers in a dictionary
        prior_distribution_modules = dict(self.model.layers.named_children())

        ## We need to fixed the names of the modules in the prior_distribution
        ## to match with the ones from the net (MLP). We need just to append
        ## 'layers.' to all the keys in the dictionary
        modules_names = copy.deepcopy(list(prior_distribution_modules.keys()))
        for key in modules_names:
            new_key = 'layers.%s' % key
            prior_distribution_modules[new_key] = prior_distribution_modules.pop(key)

        ## Get also the output layers
        prior_distribution_modules['output_layer'] = self.model.output_layer

        ## Now, get all the modules with the same names as before (and discard the others)
        net_modules = dict(map(lambda nm: nm if nm[0] in prior_distribution_modules.keys() else ('del', None), net.named_modules()))
        del net_modules['del']

        logp = 0.
        for name, module in net_modules.items():  # type: (str, nn.Linear)
            ## Now, fixing the bias in one tensor
            ## TODO: here we should check if the bias exists or not.
            ##       The next 3 lines might fail if the bias doesn't exist
            W = module.W
            b = module.b.reshape(1, -1)
            W = torch.cat([b, W], dim=0) 
            ## NOTE: the biases are in the first row of the combined weights
            ## see function: 
            ## NormFlowsLinearReparameterization.sample_predict in nflows_reparam_linear.py

            ## Finally, compute the logdensity
            logp += prior_distribution_modules[name].W_prior.logdensity(W)

        return logp


class OptimGaussianPrior(PriorModule):
    """Class of Gaussian Prior module whose parameters are optimized."""
    def __init__(self, saved_path, device="cpu"):
        """Initialization.

        Args:
            saved_path: str, the path to the checkpoint containing optimized
                parameters for Gaussian Prior.
        """
        super(OptimGaussianPrior, self).__init__()
        self.params = {}
        self.device = device

        data = torch.load(saved_path, map_location=torch.device(self.device))
        for name, param in data.items():
            self.params[name] = param.to(self.device)

    def to(self, device):
        """Move the prior's parameters to configured device.
        """
        for name in self.params.keys():
            self.params[name] = self.params[name].to(device)
        return self

    def sample(self, name, param):
        """Sample parameters from prior.

        Args:
            name: str, the name of the parameter.
            param: torch.Parameter, the parameter need to be sampled.
        """
        mu, std = self._get_params_by_name(name)

        if std is None:
            return None

        if mu is None:
            return std * torch.randn_like(param)
        else:
            return mu + std * torch.randn_like(param)

    def _get_params_by_name(self, name):
        """Get the paramters of prior by name."""

        if not (('.W' in name) or ('.b' in name)):
            return None, None
        mu, std = None, None

        if '.W' in name:
            if name.replace('.W', '.W_std') in self.params.keys():
                std = F.softplus(self.params[name.replace('.W', '.W_std')])
            if name.replace('.W', '.W_mu') in self.params.keys():
                mu = self.params[name.replace('.W', '.W_mu')]
        elif '.b' in name:
            if name.replace('.b', '.b_std') in self.params.keys():
                std = F.softplus(self.params[name.replace('.b', '.b_std')])
            if name.replace('.b', '.b_mu') in self.params.keys():
                mu = self.params[name.replace('.b', '.b_mu')]

        return mu, std

    def logp(self, net):
        """Compute the log likelihood

        Args:
            net: nn.Module, the input network needs to be evaluated.
        """
        res = 0.
        for name, param in net.named_parameters():
            mu, std = self._get_params_by_name(name)
            if std is None:
                continue
            assert std.device == param.device, "Must reconfigure the device for prior to {}".format(param.device)

            var = std ** 2
            if mu is None:
                res -= torch.sum(((param) ** 2) / (2 * var))
            else:
                res -= torch.sum(((param - mu) ** 2) / (2 * var))
        return res
