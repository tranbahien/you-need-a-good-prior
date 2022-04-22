# Copyright 2016 James Hensman, alexggmatthews, PabloLeon, Valentine Svensson
# Copyright 2017 Thomas Viehmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch


def hermgauss(n, dtype=torch.float32):
    x, w = np.polynomial.hermite.hermgauss(n)
    x, w = torch.as_tensor(x, dtype=dtype), torch.as_tensor(w, dtype=dtype)
    return x, w


def mvhermgauss(H, D, dtype=torch.float32):
    """
    Return the evaluation locations 'xn', and weights 'wn' for a multivariate
    Gauss-Hermite quadrature.

    The outputs can be used to approximate the following type of integral:
    int exp(-x)*f(x) dx ~ sum_i w[i,:]*f(x[i,:])

    Args:
        H: Number of Gauss-Hermite evaluation points.
        D: Number of input dimensions. Needs to be known at call-time.

    Returns:
        eval_locations 'x' ((H**D)xD), weights 'w' (H**D)
    """
    gh_x, gh_w = hermgauss(H)
    x = np.array(np.meshgrid(*(D*(gh_x,))))
    w = np.array(np.meshgrid(*(D*(gh_w,)))).prod(1)
    x, w = torch.as_tensor(x, dtype=dtype), torch.as_tensor(w, dtype=dtype)
    return x, w


def mvnquad(f, means, covs, H, Din, Dout=()):
    """
    Computes N Gaussian expectation integrals of a single function 'f'
    using Gauss-Hermite quadrature.

    Args:
        f: integrand function. Takes one input of shape ?xD.
        means: NxD
        covs: NxDxD
        H: Number of Gauss-Hermite evaluation points.
        Din: Number of input dimensions. Needs to be known at call-time.
        Dout: Number of output dimensions. Defaults to (). Dout is assumed
            to leave out the item index, i.e. f actually maps (?xD)->(?x*Dout).

    Returns:
        quadratures (N,*Dout)
    """
    xn, wn = mvhermgauss(H, Din)
    N = means.size(0)

    # Transform points based on Gaussian parameters
    Xt = []
    for c in covs:
        chol_cov = torch.potrf(c, upper=False) # DxD each
        Xt.append(torch.matmul(chol_cov, xn.t()))
    Xt = torch.stack(Xt, dim=0) # NxDx(H**D)
    X = 2.0 ** 0.5 * Xt + means.unsqueeze(2)  # NxDx(H**D)
    Xr = X.permute(2, 0, 1).view(-1, Din)  # (H**D*N)xD

    # Perform quadrature
    fX = f(Xr).view(*((H ** Din, N,) + Dout))
    wr = (wn * float(np.pi) ** (-Din * 0.5)).view(*((-1,) + (1,) * (1 + len(Dout))))
    return (fX * wr).sum(0)
