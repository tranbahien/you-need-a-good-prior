"""Bayesian Neural Network for classification."""

import torch

from tqdm import tqdm

from .bayes_net import BayesNet
from ..metrics.metrics_tensor import nll, accuracy
from ..utils.util import get_all_data


class ClassificationNet(BayesNet):
    def __init__(self, net, likelihood, prior, ckpt_dir, temperature=1.0,
                 sampling_method="adaptive_sghmc", logger=None, n_gpu=1):
        """Bayesian Neural Networks for classification task.

        Args:
            net: instance of BaseModel, the base neural net.
            likelihood: instance of LikelihoodModule, the module of likelihood.
            prior: instance of PriorModule, the module of prior.
            ckpt_dir: str, path to the directory of checkpoints.
            temperature: float, temperature in the posterior.
            sampling_method: specifies the sampling strategy.
            logger: the logger.
            n_gpu: int, the number of used GPUs.
        """
        BayesNet.__init__(self, net, likelihood, prior, ckpt_dir, temperature,
                          sampling_method, weights_format="state_dict",
                          task="classification", logger=logger, n_gpu=n_gpu)

    def train_and_evaluate(self, data_loader, valid_data_loader,
                           num_samples=1000, keep_every=100, lr=1e-2,
                           mdecay=0.05, batch_size=20,
                           n_discarded=0, num_burn_in_steps=3000,
                           validate_every_n_samples=10, print_every_n_samples=5,
                           resample_prior_every=1000, epsilon=1e-10,
                           resample_hyper_prior_burn_in=True,
                           continue_training=False):
        """
        Train and validates the neural network

        Args:
            data_loader: instance of DataLoader, the dataloader for training
                data.
            valid_data_loader: instance of DataLoader, the data loader for
                validation data.
            num_samples: int, number of set of parameters we want to sample.
            keep_every: number of sampling steps (after burn-in) to perform
                before keeping a sample.
            n_discarded: int, the number of first samples will
                be discarded.
            num_burn_in_steps: int, number of burn-in steps to perform.
                This value is passed to the given `sampler` if it supports
                special burn-in specific behavior like that of Adaptive SGHMC.
            lr: float, learning rate.
            batch_size: int, batch size.
            resample_prior_every: int, num ber of sampling steps to perform
                before resampling prior.
            epsilon: float, epsilon for numerical stability.
            mdecay: float, momemtum decay.
            continue_training: bool, defines whether we want to continue
                from the last training run.
        """
        if not continue_training:
            # Burn-in steps
            self.print_info("Burn-in steps")
            self.train(data_loader=data_loader, lr=lr, epsilon=epsilon,
                       mdecay=mdecay, num_burn_in_steps=num_burn_in_steps,
                       resample_prior_every=resample_prior_every,
                       resample_hyper_prior_burn_in=resample_hyper_prior_burn_in)

        _, valid_targets = get_all_data(valid_data_loader)
        valid_targets = valid_targets.to(self.device)

        best_nll = None
        preds = [] # List containing predictions
        self.print_info("Start sampling")
        for i in range(num_samples // validate_every_n_samples):
            if n_discarded > 0:
                n_discarded = 0
            self.train(data_loader=data_loader, num_burn_in_steps=0,
                       num_samples=validate_every_n_samples,
                       batch_size=batch_size,
                       lr=lr, epsilon=epsilon, mdecay=mdecay,
                       resample_hyper_prior_burn_in=resample_hyper_prior_burn_in,
                       keep_every=keep_every, continue_training=True,
                       print_every_n_samples=print_every_n_samples,
                       n_discarded=n_discarded)
            self._save_sampled_weights()

            # Make predictions
            _, new_preds = self.evaluate(valid_data_loader)
            preds.append(new_preds)

            # Evaluate the sampled weights on validation data
            mean_preds = torch.cat(preds, dim=0).mean(axis=0)
            nll_ = nll(mean_preds, valid_targets)
            accuracy_ = accuracy(mean_preds, valid_targets)

            # Save the best checkpoint
            if (best_nll is None) or (nll_ <= best_nll):
                best_nll = nll_
                self._save_checkpoint(mode="best")

            self.print_info("Validation: NLL = {:.5f} Acc = {:.4f}".format(
                nll_, accuracy_))
            self._save_checkpoint(mode="last")

            # Clear the cached weights
            self.sampled_weights.clear()

        self.print_info("Finish")

    def evaluate(self, test_data_loader, return_individual_predictions=True,
                 all_sampled_weights=False, return_label=False):
        """Evaluate the sampled weights on a given test dataset.

        Args:
            test_data_loader: instance of data loader, the data loader for
                test data.
            return_individual_predictions: bool, if True also the predictions
                of the individual models are returned.
            all_sampled_weights: bool, if True load all sampled weights from
                file to make predictions; otherwise, only use the current
                sampled weights in the lists `self.sampled_weights`.
            return_label: bool, if True, return the labels along with the
                data.
        """
        self.net.eval()

        def network_predict(test_data_loader_, weights):
            predictions = []
            targets = []
            with torch.no_grad():
                self.network_weights = weights

                for x_batch, y_batch in test_data_loader_:
                    x_batch = x_batch.to(self.device)
                    predictions.append(self.net.predict(x_batch).float())
                    targets.append(y_batch)
                
                return torch.cat(predictions, dim=0), torch.cat(targets, dim=0)

        # Make predictions
        predictions = []
        targets = None

        if all_sampled_weights:
            sampled_weights_loader = self._load_all_sampled_weights()
            for weights in tqdm(sampled_weights_loader):
                preds, targets = network_predict(test_data_loader,
                                                 weights=weights)
                predictions.append(preds)
        else:
            for weights in self.sampled_weights:
                preds, targets = network_predict(test_data_loader,
                                                 weights=weights)
                predictions.append(preds)

        predictions = torch.stack(predictions, dim=0)

        # Estimate the predictive mean
        mean_predictions = torch.mean(predictions, axis=0)
        mean_predictions = mean_predictions.to(self.device)

        if return_individual_predictions:
            if return_label:
                return mean_predictions, predictions, targets
            else:
                return mean_predictions, predictions

        return mean_predictions

    def predict(self, x_test, all_sampled_weights=True,
                return_individual_predictions=False,
                num_samples=None):
        """Evaluate the sampled weights on a given test dataset.

        Args:
            x_test: torch tensor, the test datapoint.
            all_sampled_weights: bool, if True loadd all sampled weights from
                file to make predictions; otherwise, only use the current
                sampled weights in the lists `self.sampled_weights`.
            return_individual_predictions: bool, if True also the predictions
                of the individual models are returned.
            num_samples: int, the number of sampled used for testing.

        Returns:
            torch tensor, the predicted mean.
        """
        self.net.eval()

        def network_predict(x_test_, weights, device):
            with torch.no_grad():
                self.network_weights = weights
                return self.net.predict(x_test_.to(device))

        # Make predictions
        network_outputs = []

        if all_sampled_weights:
            sampled_weights_loader = self._load_all_sampled_weights()
            for idx, weights in enumerate(sampled_weights_loader):
                network_outputs.append(network_predict(
                    x_test, weights, self.device))
                if num_samples is not None:
                    if (idx+1) > num_samples:
                        break
        else:
            for weights in self.sampled_weights:
                network_outputs.append(network_predict(
                    x_test, weights, self.device))

        predictions = torch.stack(network_outputs, dim=0)

        # Estimate the predictive mean
        mean_predictions = torch.mean(predictions, axis=0)
        mean_predictions = mean_predictions.to(self.device)

        if return_individual_predictions:
            return mean_predictions, predictions

        return mean_predictions

    def _print_evaluations(self, x, y, train=True):
        """Evaluate the sampled weights on training/validation data and
            during the training log the results.

        Args:
            x: numpy array, shape [batch_size, num_features], the input data.
            y: numpy array, shape [batch_size, 1], the corresponding targets.
            train: bool, indicate whether we're evaluating on the training data.
        """
        preds = self.predict(x, all_sampled_weights=(not train))
        acc_ = accuracy(preds, y)
        nll_ = nll(preds, y)

        if train:
            self.print_info("Samples # {:5d} : NLL = {:.5f} "
                "Acc = {:.4f} ".format(self.num_samples, nll_, acc_))
        else:
            self.print_info("Validation: NLL = {:.5f} Acc = {:.4f}".format(
                nll_, acc_))
