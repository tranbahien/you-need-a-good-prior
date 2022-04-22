import torch
import torch.nn as nn
import numpy as np


class FactorizedGaussianDistribution(nn.Module):
    def __init__(self, *shape):
        super(FactorizedGaussianDistribution, self).__init__()
        self.shape = shape
        self.mean = nn.Parameter(torch.zeros(*shape))
        self.logvars = nn.Parameter(torch.zeros(*shape))

    def sample(self, *shape):
        epsilon_for_samples = torch.randn(*shape, *self.shape,
                                          device=self.mean.device,
                                          requires_grad=False)

        samples = self.mean + (self.logvars/2).exp() * epsilon_for_samples
        return samples
    
    def logdensity(self, inputs):
        return torch.sum(-0.5 * (np.log(2 * np.pi) + self.logvars + (self.mean - inputs) ** 2 / self.logvars.exp()))


class SinglePlanarFlow(nn.Module):
    def __init__(self, d):
        super(SinglePlanarFlow, self).__init__()
        self.d = d
        self.u = nn.Parameter(torch.randn(1, d) / np.sqrt(d))
        self.w = nn.Parameter(torch.randn(1, d) / np.sqrt(d))
        self.b = nn.Parameter(torch.randn(1))
        self.h_fn = lambda x: torch.tanh(x)
        self.h_prime_fn = lambda x: 1 - torch.tanh(x) ** 2
        self.log_det_jacobian = torch.zeros(1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.w, -np.sqrt(1/self.d), np.sqrt(1/self.d))
        torch.nn.init.uniform_(self.u, -np.sqrt(1/self.d), np.sqrt(1/self.d))
        torch.nn.init.uniform_(self.b, -np.sqrt(1/self.d), np.sqrt(1/self.d))

        
    def forward(self, x):
        u_hat = self.u
        if self.w @ self.u.t() < 1:
            wtu = (self.w @ self.u.t()).squeeze()
            m_wtu = - 1 + torch.log1p(wtu.exp())
            u_hat = self.u + (m_wtu - wtu) * self.w / (self.w @ self.w.t())
        
        lin = x @ self.w.t() + self.b
        out = x + self.h_fn(lin) @ u_hat
        
        psi = self.h_prime_fn(lin) * self.w
            
        
        self.log_det_jacobian = torch.log(torch.abs(1 + psi @ u_hat.t()))
        
        return out
    
    
class NormalizingFlow(nn.Module):
    def __init__(self, *shape, n_transformations=32, flow_class=SinglePlanarFlow):
        super(NormalizingFlow, self).__init__()
        self.n_transformations = n_transformations
        self.shape = shape
        self.vshape = np.prod(shape)   # vectorized shape
        self.initial_distribution = FactorizedGaussianDistribution(self.vshape)
        self.initial_distribution.mean.requires_grad = False
        self.flows = nn.Sequential(*[flow_class(self.vshape) for _ in range(n_transformations)])

    def sample(self, *shape):
        q0 = self.initial_distribution.sample(*shape)
        qk = self.flows(q0)
        return qk.reshape(*shape, *self.shape)
    
    def extra_repr(self):
        return "shape=%s" % str(self.shape)

    def logdensity(self, x):
        x = x.reshape(-1, self.vshape)
        return self.initial_distribution.logdensity(self.flows(x)) + torch.sum(torch.tensor([flow.log_det_jacobian for flow in self.flows]))

