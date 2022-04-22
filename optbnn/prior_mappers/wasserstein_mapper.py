import numpy as np
import torch
import torch.nn as nn
import os

import itertools
from torch.utils.data import TensorDataset, DataLoader

from ..utils.util import ensure_dir, prepare_device


class LipschitzFunction(nn.Module):
    def __init__(self, dim):
        super(LipschitzFunction, self).__init__()
        self.lin1 = nn.Linear(dim, 200)
        self.relu1 = nn.Softplus()
        self.lin2 = nn.Linear(200, 200)
        self.relu2 = nn.Softplus()
        self.lin3 = nn.Linear(200, 1)

    def forward(self, x):
        x = x.float()
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        x = self.relu2(x)
        x = self.lin3(x)
        return x


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)


class WassersteinDistance():
    def __init__(self, bnn, gp, lipschitz_f_dim,
                 output_dim, use_lipschitz_constraint=True,
                 lipschitz_constraint_type="gp", wasserstein_lr=0.01,
                 device='cpu', gpu_gp=True):
        self.bnn = bnn
        self.gp = gp
        self.device = device
        self.output_dim = output_dim
        self.lipschitz_f_dim = lipschitz_f_dim
        self.lipschitz_constraint_type = lipschitz_constraint_type
        assert self.lipschitz_constraint_type in ["gp", "lp"]

        self.lipschitz_f = LipschitzFunction(dim=lipschitz_f_dim)
        self.lipschitz_f = self.lipschitz_f.to(self.device)
        self.gpu_gp = gpu_gp
        self.values_log = []

        self.optimiser = torch.optim.Adagrad(self.lipschitz_f.parameters(),
                                             lr=wasserstein_lr)
        self.use_lipschitz_constraint = use_lipschitz_constraint
        self.penalty_coeff = 10

    def calculate(self, nnet_samples, gp_samples):
        d = 0.
        for dim in range(self.output_dim):
            f_samples = self.lipschitz_f(nnet_samples[:, :, dim].T)
            f_gp = self.lipschitz_f(gp_samples[:, :, dim].T)
            d += torch.mean(torch.mean(f_samples, 0) - torch.mean(f_gp, 0))
        return d

    def compute_gradient_penalty(self, samples_p, samples_q):
        eps = torch.rand(samples_p.shape[1], 1).to(samples_p.device)
        X = eps * samples_p.t().detach() + (1 - eps) * samples_q.t().detach()
        X.requires_grad = True
        Y = self.lipschitz_f(X)
        gradients = torch.autograd.grad(
            Y, X, grad_outputs=torch.ones(Y.size(), device=self.device),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        f_gradient_norm = gradients.norm(2, dim=1)

        if self.lipschitz_constraint_type == "gp":
            # Gulrajani2017, Improved Training of Wasserstein GANs
            return ((f_gradient_norm - 1) ** 2).mean()

        elif self.lipschitz_constraint_type == "lp":
            # Henning2018, On the Regularization of Wasserstein GANs
            # Eq (8) in Section 5
            return ((torch.clamp(f_gradient_norm - 1, 0., np.inf))**2).mean()

    def wasserstein_optimisation(self, X, n_samples, n_steps=10, threshold=None, debug=False):
        for p in self.lipschitz_f.parameters():
            p.requires_grad = True

        n_samples_bag = n_samples * 1
        if not self.gpu_gp:
            X = X.to("cpu")

        # Draw functions from GP
        gp_samples_bag = self.gp.sample_functions(
            X.double(), n_samples_bag).detach().float().to(self.device)
        if self.output_dim > 1:
            gp_samples_bag = gp_samples_bag.squeeze()

        if not self.gpu_gp:
            X = X.to(self.device)

        # Draw functions from Bayesian Neural network
        nnet_samples_bag = self.bnn.sample_functions(
            X, n_samples_bag).detach().float().to(self.device)
        if self.output_dim > 1:
            nnet_samples_bag = nnet_samples_bag.squeeze()

        #  It was of size: [n_dim, N, n_out]
        # will be of size: [N, n_dim, n_out]
        gp_samples_bag = gp_samples_bag.transpose(0, 1)
        nnet_samples_bag = nnet_samples_bag.transpose(0, 1)
        dataset = TensorDataset(gp_samples_bag, nnet_samples_bag)
        data_loader = DataLoader(dataset, batch_size=n_samples, num_workers=0)
        batch_generator = itertools.cycle(data_loader)

        for i in range(n_steps):
            gp_samples, nnet_samples = next(batch_generator)
            #         was of size: [N, n_dim, n_out]
            # needs to be of size: [n_dim, N, n_out]
            gp_samples = gp_samples.transpose(0, 1)
            nnet_samples = nnet_samples.transpose(0, 1)

            self.optimiser.zero_grad()
            objective = -self.calculate(nnet_samples, gp_samples)
            if debug:
                self.values_log.append(-objective.item())

            if self.use_lipschitz_constraint:
                penalty = 0.
                for dim in range(self.output_dim):
                    penalty += self.compute_gradient_penalty(
                        nnet_samples[:, :, dim], gp_samples[:, :, dim])
                objective += self.penalty_coeff * penalty
            objective.backward()

            if threshold is not None:
                # Gradient Norm
                params = self.lipschitz_f.parameters()
                grad_norm = torch.cat([p.grad.data.flatten() for p in params]).norm()

            self.optimiser.step()
            if not self.use_lipschitz_constraint:
                for p in self.lipschitz_f.parameters():
                    p.data = torch.clamp(p, -.1, .1)
            if threshold is not None and grad_norm < threshold:
                print('WARNING: Grad norm (%.3f) lower than threshold (%.3f). ', end='')
                print('Stopping optimization at step %d' % (i))
                if debug:
                    ## '-1' because the last wssr value is not recorded
                    self.values_log = self.values_log + [self.values_log[-1]] * (n_steps-i-1)
                break
        for p in self.lipschitz_f.parameters():
            p.requires_grad = False


class MapperWasserstein(object):
    def __init__(self, gp, bnn, data_generator, out_dir,
                 input_dim=1, output_dim=1, n_data=256,
                 wasserstein_steps=(200, 200), wasserstein_lr=0.01, wasserstein_thres=0.01, 
                 logger=None, n_gpu=0, gpu_gp=False, lipschitz_constraint_type="gp"):
        self.gp = gp
        self.bnn = bnn
        self.data_generator = data_generator
        self.n_data = n_data
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.out_dir = out_dir
        self.device, device_ids = prepare_device(n_gpu)
        self.gpu_gp = gpu_gp

        assert lipschitz_constraint_type in ["gp", "lp"]
        self.lipschitz_constraint_type = lipschitz_constraint_type

        if type(wasserstein_steps) != list and type(wasserstein_steps) != tuple:
            wasserstein_steps = (wasserstein_steps, wasserstein_steps)
        self.wasserstein_steps = wasserstein_steps
        self.wasserstein_threshold = wasserstein_thres

        # Move models to configured device
        if gpu_gp:
            self.gp = self.gp.to(self.device)
        self.bnn = self.bnn.to(self.device)
        if len(device_ids) > 1:
            if self.gpu_gp:
                self.gp = torch.nn.DataParallel(self.gp, device_ids=device_ids)
            self.bnn = torch.nn.DataParallel(self.bnn, device_ids=device_ids)

        # Initialize the module of wasserstance distance
        self.wasserstein = WassersteinDistance(
            self.bnn, self.gp,
            self.n_data, output_dim=self.output_dim,
            wasserstein_lr=wasserstein_lr, device=self.device,
            gpu_gp=self.gpu_gp,
            lipschitz_constraint_type=self.lipschitz_constraint_type)

        # Setup logger
        self.print_info = print if logger is None else logger.info

        # Setup checkpoint directory
        self.ckpt_dir = os.path.join(self.out_dir, "ckpts")
        ensure_dir(self.ckpt_dir)

    def optimize(self, num_iters, n_samples=128, lr=1e-2,
                 save_ckpt_every=50, print_every=10, debug=False):
        wdist_hist = []

        wasserstein_steps = self.wasserstein_steps
        prior_optimizer = torch.optim.RMSprop(self.bnn.parameters(), lr=lr)

        # Prior loop
        for it in range(1, num_iters+1):
            # Draw X
            X = self.data_generator.get(self.n_data)
            X = X.to(self.device)
            if not self.gpu_gp:
                X = X.to("cpu")

            # Draw functions from GP
            gp_samples = self.gp.sample_functions(
                X.double(), n_samples).detach().float().to(self.device)
            if self.output_dim > 1:
                gp_samples = gp_samples.squeeze()

            if not self.gpu_gp:
                X = X.to(self.device)

            # Draw functions from BNN
            nnet_samples = self.bnn.sample_functions(
                X, n_samples).float().to(self.device)
            if self.output_dim > 1:
                nnet_samples = nnet_samples.squeeze()

            ## Initialisation of lipschitz_f
            self.wasserstein.lipschitz_f.apply(weights_init)

            # Optimisation of lipschitz_f
            self.wasserstein.wasserstein_optimisation(X, 
                n_samples, n_steps=wasserstein_steps[1],
                threshold=self.wasserstein_threshold, debug=debug)
            prior_optimizer.zero_grad()


            wdist = self.wasserstein.calculate(nnet_samples, gp_samples)
            wdist.backward()
            prior_optimizer.step()

            wdist_hist.append(float(wdist))
            if (it % print_every == 0) or it == 1:
                self.print_info(">>> Iteration # {:3d}: "
                                "Wasserstein Dist {:.4f}".format(
                                    it, float(wdist)))

            # Save checkpoint
            if ((it) % save_ckpt_every == 0) or (it == num_iters):
                path = os.path.join(self.ckpt_dir, "it-{}.ckpt".format(it))
                torch.save(self.bnn.state_dict(), path)

        # Save accumulated list of intermediate wasserstein values
        if debug:
            values = np.array(self.wasserstein.values_log).reshape(-1, 1)
            path = os.path.join(self.out_dir, "wsr_intermediate_values.log")
            np.savetxt(path, values, fmt='%.6e')
            self.print_info('Saved intermediate wasserstein values in: ' + path)

        return wdist_hist
