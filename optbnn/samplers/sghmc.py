import torch
import numpy as np
from torch.optim import Optimizer


class SGHMC(Optimizer):
    """ Stochastic Gradient Hamiltonian Monte-Carlo Sampler.

        References:
        [1] https://arxiv.org/pdf/1402.4102.pdf
    """
    name = "SGHMC"

    def __init__(self, params, lr=1e-2, mdecay=0.05, scale_grad=1.):
        """ Set up a SGHMC Optimizer.

        Args:
            params: iterable, parameters serving as optimization variable.
            lr: float, base learning rate for this optimizer.
                Must be tuned to the specific function being minimized.
            mdecay:float, momentum decay per time-step.
            scale_grad: float, optional
                Value that is used to scale the magnitude of the epsilon used
                during sampling. In a typical batches-of-data setting this
                usually corresponds to the number of examples in the
                entire dataset.
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(
            lr=lr, scale_grad=scale_grad,
            mdecay=mdecay
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]

                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.zeros_like(parameter,
                                                         dtype=parameter.dtype)

                state["iteration"] += 1

                mdecay, lr = group["mdecay"], group["lr"]
                scale_grad = group["scale_grad"]

                momentum = state["momentum"]
                gradient = parameter.grad.data * scale_grad

                # Sample random epsilon
                sigma = torch.sqrt(torch.from_numpy(
                    np.array(2 * lr * mdecay, dtype=type(lr))))
                sample_t = torch.normal(mean=torch.zeros_like(gradient),
                                        std=torch.ones_like(gradient) * sigma)

                # Update momentum (Eq 15 below in [1])
                momentum.add_(-lr * gradient - mdecay * momentum + sample_t)

                # Update parameter (Eq 15 above in [1])
                parameter.data.add_(momentum)

        return loss
