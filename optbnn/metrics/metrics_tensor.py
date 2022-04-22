"""A PyTorch implementation of evaluation metrics."""

import numpy as np
import torch


def nll_gaussian(output, target, sigma, no_dim=1):
    """Computes the negative log-likelihood of Deep Ensembles for regerssion
        tasks.

    Args:
        output: torch tensor, shape [batch_size, 1], the predictions of mean
            from, a Deep Ensembles model.
        target: torch tensor, shape [batch_size, 1], the corresponding label
            data.
        sigma: torch tensor, shape [batch_size, 1], the predictions of
            standard-deviation from a Deep Ensembles model.
        no_dim: int, the number dimension of label data.

    Returns:
        torch tensor, the resulting negative log-likelihood.
    """
    with torch.no_grad():
        exponent = 0.5*(target - output)**2 / sigma**2
        log_coeff = no_dim*torch.log(sigma) + 0.5*no_dim*np.log(2*np.pi)

    return (log_coeff + exponent).mean()


def nll_mc_dropout(preds, target, T, tau):
    """Computes the negative log-likelihood of MC-dropout models for
        regression tasks.

    Args:
        preds: torch tensor, shape [batch_size, T, 1], the predictions of
            MC-Dropout model with T forward passes on a batch of B examples.
        target: torch tensor, shape[batch_size, 1], the corresponding label
            data.
        T: int, the number forward passes for predictions.
        tau: float, the precision hyper-parameter of MC-dropout model.

    Returns:
        1D torch tensor -- The resulting negative log likelihood.
    """
    with torch.no_grad():
        preds = preds.view(preds.shape[0], preds.shape[1], 1)
        target = target.view(-1, 1)

        ll = torch.logsumexp(-0.5 * tau * (target[None] - preds)**2., 0) - \
            torch.log(torch.scalar_tensor(T)) - \
            0.5 * torch.log(torch.scalar_tensor(2*np.pi)) + \
            0.5 * torch.log(torch.scalar_tensor(tau))

    return -torch.mean(ll)


def rmse(output, target):
    """Computes the Root Mean Squared Error for regression tasks.

    Args:
        output: torch tensor, shape [batch_size, 1], the predictions.
        target: torch tensor, shape [batch_size, 1], the corresponding targets.

    Returns:
        torch tensor: the resulting RMSE.
    """
    with torch.no_grad():
        result = torch.mean((output.squeeze() - target.squeeze())**2.)**0.5
    return result


def nll(output, target):
    """Computes the negative log-likelihood for classification problem
        (cross-entropy).

    Args:
        output: float torch tensor, shape [batch_size, num_classes],
            The predictions of probabilites for each classes
        target: int torch tensor, shape [batch_size, 1], the corresponding
            labels.

    Returns:
        torch tensor: the resulting negative log-likelihood.
    """
    with torch.no_grad():
        result = -torch.log(output)[range(target.shape[0]), target].mean()
    return result


def brier_score(output, target):
    """Computes the Brier score used to evaluate the quality of model
        uncertainty.

    Args:
        output: float torch tensor, shape [batch_size, num_classes],
            The predictions of probabilites for each classes
        target: int torch tensor, shape [batch_size, 1], the corresponding
            labels.

    Returns:
        torch tensor: the resulting Brier score.
    """
    with torch.no_grad():
        num_classes = output.shape[1]

        targets_one_hot = torch.zeros_like(output)
        targets_one_hot[torch.arange(target.shape[0]), target] = 1.

        squared_diff = (targets_one_hot - output) ** 2
        score = torch.mean(torch.div(torch.sum(squared_diff, axis=1),
                                     num_classes))
    return score


def entropy(output, target=None):
    """Compute the Entropy of predictions.

    Args:
        output: float torch tensor, shape [batch_size, num_classes],
            The predictions of probabilites for each classes
        target: int torch tensor, shape [batch_size, 1], the corresponding
            labels.

    Returns:
        torch tensor: the resulting entropy.
    """
    return torch.sum(-output*torch.log(output), axis=1).mean()


def accuracy(output, target):
    """Computes the accuracy of predictions.

    Args:
        output: float torch tensor, shape [batch_size, num_classes],
            The predictions of probabilites for each classes
        target: int torch tensor, shape [batch_size, 1], the corresponding
            labels.

    Returns:
        torch tensor: the resulting accuracy.
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    """Computes the top-k accuracy of predictions.

    Args:
        output: float torch tensor, shape [batch_size, num_classes],
            The predictions of probabilites for each classes
        target: int torch tensor, shape [batch_size, 1], the corresponding
            labels.
        k: int, the top-k guesses.

    Returns:
        torch tensor: the resulting accuracy.
    """
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def mean_squared_error(output, target):
    """Computes the mean squared error of predictions.

    Args:
        output: float torch tensor, shape [batch_size, num_classes],
            The predictions of probabilites for each classes
        target: int torch tensor, shape [batch_size, 1], the corresponding
            labels.

    Returns:
        torch tensor: the resulting MSE.
    """
    with torch.no_grad():
        error = torch.sum((output - target) ** 2)
    return error / len(target)
