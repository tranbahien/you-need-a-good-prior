"""Bayesian Neural Network for regression."""

import numpy as np
import torch
import copy

from .bayes_net import BayesNet
from ..utils.normalization import zscore_normalization, zscore_unnormalization
from ..metrics.uncertainty import gaussian_nll, rmse


class RegressionNet(BayesNet):
    def __init__(self, net, likelihood, prior, ckpt_dir,
                 temperature=1.0, normalize_input=True,
                 normalize_output=True, sampling_method="adaptive_sghmc",
                 logger=None, n_gpu=0):
        """Bayesian Neural Networks for regression task.

        Args:
            net: instance of nn.Module, the base neural network.
            likelihood: instance of LikelihoodModule, the module of likelihood.
            prior: instance of PriorModule, the module of prior.
            ckpt_dir: str, path to the directory of checkpoints.
            temperature: float, the temperature in posterior.
            normalize_input: bool, defines whether to normalize the inputs.
            normalize_output: bool, defines whether to normalize the outputs.
            sampling_method: specifies the sampling strategy.
            logger: instance of logging.Logger.
            n_gpu: int, the number of used GPUs.
        """
        BayesNet.__init__(self, net, likelihood, prior, ckpt_dir, temperature,
                          sampling_method, weights_format="tuple",
                          task="regression", logger=logger, n_gpu=n_gpu)
        self.do_normalize_input = normalize_input
        self.do_normalize_output = normalize_output

    def train_and_evaluate(self, x_train, y_train, x_valid, y_valid,
                           num_samples=1000, keep_every=100, lr=1e-2,
                           mdecay=0.05, batch_size=20, num_burn_in_steps=3000,
                           validate_every_n_samples=10, print_every_n_samples=5,
                           epsilon=1e-10, continue_training=False):
        """
        Train and validates the bayesian neural network

        Args:
            x_train: numpy array, input training datapoints.
            y_train: numpy array, input training targets.
            x_valid: numpy array, input validation datapoints.
            y_valid: numpy array, input validation targets.
            num_samples: int, number of set of parameters we want to sample.
            keep_every: number of sampling steps (after burn-in) to perform
                before keeping a sample.
            lr: float, learning rate.
            mdecay: float, momemtum decay.
            batch_size: int, batch size.
            num_burn_in_steps: int, number of burn-in steps to perform.
                This value is passed to the given `sampler` if it supports
                special burn-in specific behavior like that of Adaptive SGHMC.
            validate_every_n_samples: int, defines after how many samples we
                want to evaluate the sampled weights on validation data.
            print_every_n_samples: int, defines after how many samples we want
                to evaluate the sampled weights on training data.
            epsilon: float, epsilon for numerical stability.
            continue_training: bool, defines whether we want to continue
                from the last training run.
        """
        # Burn-in steps
        self.print_info("Burn-in steps")
        self.train(x_train=x_train, y_train=y_train,
                   num_burn_in_steps=num_burn_in_steps,
                   lr=lr, epsilon=epsilon, mdecay=mdecay)

        self.print_info("Start sampling")
        for i in range(num_samples // validate_every_n_samples):
            self.train(x_train=x_train, y_train=y_train, num_burn_in_steps=0,
                       num_samples=validate_every_n_samples,
                       batch_size=batch_size,
                       lr=lr, epsilon=epsilon, mdecay=mdecay,
                       keep_every=keep_every, continue_training=True,
                       print_every_n_samples=print_every_n_samples)
            self._print_evaluations(x_valid, y_valid, False)

        self._save_sampled_weights()
        self.print_info("Finish")

    def predict(self, x_test, return_individual_predictions=False,
                return_raw_predictions=False):
        """Predicts mean and variance for the given test point.

        Args:
            x_test: numpy array, the test datapoint.
            return_individual_predictions: bool, if True also the predictions
                of the individual models are returned.
            return_raw_predictions: bool, indicates whether or not return
                the raw predictions along with the unnormalized predictions.

        Returns:
            a tuple consisting of mean and variance.
        """
        x_test_ = np.asarray(x_test).reshape(x_test.shape[0], -1)

        # Normalize the data
        if self.do_normalize_input:
            x_test_, *_ = zscore_normalization(x_test_, self.x_mean, self.x_std)

        def network_predict(x_test_, weights):
            with torch.no_grad():
                self.network_weights = weights
                return self.net(torch.from_numpy(x_test_).float().to(self.device)).\
                    detach().cpu().numpy()

        # Make predictions for each sampled weights
        predictions = np.array([
            network_predict(x_test_, weights=weights)
            for weights in self.sampled_weights])

        # Calculates the predictive mean and variance
        pred_mean = np.mean(predictions, axis=0)
        pred_var = np.var(predictions, axis=0) + self.lik_module.var

        # Unnormalize the data
        if self.do_normalize_output:
            pred_mean = zscore_unnormalization(pred_mean, self.y_mean,
                                               self.y_std)
            pred_var *= self.y_std ** 2

            if return_raw_predictions:
                raw_predictions = copy.deepcopy(predictions).squeeze()

            for i in range(len(predictions)):
                predictions[i] = zscore_unnormalization(
                    predictions[i], self.y_mean, self.y_std)

        if return_individual_predictions:
            if return_raw_predictions:
                return pred_mean, pred_var, predictions, raw_predictions
            else:
                return pred_mean, pred_var, predictions

        return pred_mean, pred_var

    def _print_evaluations(self, x, y, train=True):
        """Evaluate the sampled weights on training/validation data and
            during the training log the results.

        Args:
            x: numpy array, shape [batch_size, num_features], the input data.
            y: numpy array, shape [batch_size, 1], the corresponding targets.
            train: bool, indicate whether we're evaluating on the training data.
        """
        self.net.eval()
        pred_mean, pred_var = self.predict(x)
        total_nll = gaussian_nll(y, pred_mean, pred_var)
        total_rmse = rmse(pred_mean, y)

        if train:
            self.print_info("Samples # {:5d} : NLL = {:11.4e} "
                 "RMSE = {:.4e} ".format(self.num_samples, total_nll,
                                         total_rmse))
        else:
            self.print_info("Validation: NLL = {:11.4e} RMSE = {:.4e}".format(
                total_nll, total_rmse))

        self.net.train()

    def _normalize_data(self, x_train, y_train):
        """Normalize the training data and save the resulting statistics.

        Args:
            x_train: numpy array, input training datapoints.
            y_train: numpy array, input training targets.

        Returns:
            x_train_: numpy array, the normalized training datapoints.
            y_train_: numpy array, the normalized training targets.
        """
        if self.do_normalize_input:
            x_train_, self.x_mean, self.x_std = zscore_normalization(x_train)
            x_train_ = torch.from_numpy(x_train_).float()
        else:
            x_train_ = torch.from_numpy(x_train).float()

        if self.do_normalize_output:
            y_train_, self.y_mean, self.y_std = zscore_normalization(y_train)
            y_train_ = torch.from_numpy(y_train_).float()
        else:
            y_train_ = torch.from_numpy(y_train).float()

        return x_train_, y_train_

        
