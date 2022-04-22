import sys
import torch

from . import util

from itertools import islice


class ClassificationGenerator(object):
    def __init__(self, data_loader):
        self.batch_size = data_loader.batch_size
        self.data_iterator = islice(enumerate(util.inf_loop(data_loader)),
                                    sys.maxsize)

    def get(self, n_data=None, return_label=False):
        _, data = next(self.data_iterator)
        X, y = data[0], data[1]

        # Retry to make sure to have the right mini-batch size
        if n_data is not None:
            if X.shape[0] != n_data:
                _, data = next(self.data_iterator)
                X, y = data[0], data[1]
                assert X.shape[0] == n_data

        if return_label:
            return X, y
        else:
            return X


class UniformGenerator(object):
    def __init__(self, x_min, x_max, input_dim=1):
        self.x_min = x_min
        self.x_max = x_max
        self.input_dim = input_dim

        # Initialize generator to create random pointss
        if isinstance(x_max, float):
            self.rand_generator = torch.distributions.uniform.Uniform(
                x_min, x_max)
        else:
            self.rand_generator = torch.distributions.uniform.Uniform(
                torch.from_numpy(x_min).float(),
                torch.from_numpy(x_max).float())

    def get(self, n_data):
        X = self.rand_generator.rsample([n_data])
        return X.reshape([-1, self.input_dim])


class GridGenerator(object):
    def __init__(self, x_min, x_max, input_dim=1):
        self.x_min = x_min
        self.x_max = x_max
        self.input_dim = input_dim

    def get(self, n_data):
        X = torch.linspace(self.x_min, self.x_max, n_data)
        return X.reshape([-1, self.input_dim])


class MeasureSetGenerator(object):
    def __init__(self, X, x_min, x_max, real_ratio=0.8, fix_data=False):
        if not isinstance(X, torch.Tensor):
            self.X = torch.from_numpy(X).float()
        else:
            self.X = X.float()
        self.x_min = x_min
        self.x_max = x_max
        self.real_ratio = real_ratio

        # Initialize generator to create random pointss
        self.rand_generator = torch.distributions.uniform.Uniform(
            torch.from_numpy(self.x_min).float(),
            torch.from_numpy(self.x_max).float())
        
        self.use_cache = fix_data
        self.X_cached = None

    def get(self, n_data):
        if self.use_cache and self.X_cached is not None:
            return self.X_cached
        
        n_real = int(n_data * self.real_ratio)
        # assert n_real < self.X.shape[0]
        n_real = min(n_real, int(self.X.shape[0]))
        n_rand = n_data - n_real

        # Choose randomly training inputs
        indices = torch.randperm(self.X.shape[0])[:n_real]

        # Generate random points
        X_real = self.X[indices, ...]
        X_rand = self.rand_generator.rsample([n_rand])

        # Concatenate both sets
        X = torch.cat((X_real, X_rand), axis=0)
        indices = torch.randperm(X.shape[0])
        X = X[indices, ...]

        if self.use_cache:
            self.X_cached = X
        return X

