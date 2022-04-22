"""Defines metrics used to evaluate uncertainty."""

import torch
import numpy as np

from scipy.stats import norm

from ..utils.util import to_one_hot


def gaussian_nll(y, mu, var):
    """Calculates the negative log likelihood of Gaussian distribution.

    Args:
        y: numpy array, shape [batch_size], the true labels.
        mu: numpy array, shape [batch_size], the predicted means.
        var: numpy array, shape [batch_size], the predicted variances.

    Returns:
        nll: float, the resulting negative log likelihood
    """
    y, mu, var = y.squeeze(), mu.squeeze(), var.squeeze()
    nll = -np.mean(norm.logpdf(y, loc=mu, scale=np.sqrt(var)))

    return float(nll)


def rmse(y_pred, y):
    """Calculates the root mean squared error.

    Args:
        y_pred: numpy array, shape [batch_size], the predictions.
        y: numpy array, shape [batch_size], the corresponding labels.

    Returns:
        rmse: float, the resulting root mean squared error.
    """
    y_pred, y = y_pred.squeeze(), y.squeeze()
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    return float(rmse)


def compute_regression_calibration(pred_mean, pred_var, target, num_bins=10):
    """Compute the regression calibration. Note that we assume that the
        probabilistic forecase taking the form of Gaussian.
        References:
        [1] https://arxiv.org/abs/1807.00263

    Args:
        pred_mean: numpy array, shape [num_data, ], the predicted mean.
        pred_var: numpy array, shape [num_data, ], the predicted variance.
        target: numpy array, shape [num_data, ], the ground truths.
        num_bins: number of bins.

    Returns:
        cal: a dictionary
            {reliability_diag: realibility diagram
            calibration_error: calibration error,
            sharpness: sharpness
            }
    """
    # Make sure the inputs have valid shape
    pred_mean = pred_mean.flatten()
    pred_var = pred_var.flatten()
    target = target.flatten()

    # Compute the predicted CDF
    predicted_cdf = norm.cdf(target, loc=pred_mean, scale=np.sqrt(pred_var))
    # Compute the empirical CDF
    # empirical_cdf = np.zeros(len(predicted_cdf))
    # for i, p in enumerate(predicted_cdf):
    #     empirical_cdf[i] = np.mean(predicted_cdf <= p)

    # Initialize the expected confidence levels according to the number of bins
    expected_conf_levels = np.linspace(0, 1, num_bins+1)[1:]

    # Compute the observed confidence levels, Eq (8) in [1].
    observed_conf_levels = np.zeros_like(expected_conf_levels)
    for i, p in enumerate(expected_conf_levels):
        observed_conf_levels[i] = np.mean(predicted_cdf < p)

    # Compute the calibration error, Eq (9) in [1].
    calibration_error = float(np.sum((expected_conf_levels -
                                      observed_conf_levels)**2))

    # Compute the sharpness of the predictions, Eq (10) in [1].
    sharpness = np.mean(pred_var)

    # Repliability diagram
    reliability_diag = {
        "expected_conf_levels": expected_conf_levels,
        "observed_conf_levels": observed_conf_levels
    }

    # Saving
    cal = {
        'reliability_diag': reliability_diag,
        'calibration_error': calibration_error,
        'sharpness': sharpness
    }

    return cal


def filter_top_k(probabilities, labels, top_k):
    """Extract top k predicted probabilities and corresponding ground truths"""
    labels_one_hot = np.zeros(probabilities.shape)
    labels_one_hot[np.arange(probabilities.shape[0]), labels] = 1

    if top_k is None:
        return probabilities, labels_one_hot

    negative_prob = -1. * probabilities

    ind = np.argpartition(negative_prob, top_k-1, axis=-1)
    top_k_ind = ind[:, :top_k]
    rows = np.expand_dims(np.arange(probabilities.shape[0]), axis=1)
    lowest_k_negative_probs = negative_prob[rows, top_k_ind]
    output_probs = -1. * lowest_k_negative_probs

    labels_one_hot_k = labels_one_hot[rows, top_k_ind]

    return output_probs, labels_one_hot_k


def get_multiclass_predictions_and_correctness(probabilities, labels, top_k=1):
    """Returns predicted class, correctness boolean vector."""
    if top_k == 1:
        class_predictions = np.argmax(probabilities, -1)
        top_k_probs = probabilities[np.arange(len(labels)), class_predictions]
        is_correct = np.equal(class_predictions, labels)
    else:
        top_k_probs, is_correct = filter_top_k(probabilities, labels, top_k)

    return top_k_probs, is_correct


def accuracy(probabilities, labels):
    """Computes the top-1 accuracy of predictions.

    Args:
        probabilities: Array of probabilities of shape 
            [num_samples, num_classes].
        labels: Integer array labels of shape [num_samples].
    Returns:
        float: Top-1 accuracy of predictions.
    """
    return accuracy_top_k(probabilities, labels, 1)


def accuracy_top_k(probabilities, labels, top_k):
    """Computes the top-k accuracy of predictions.

    Args:
        probabilities: Array of probabilities of shape 
            [num_samples, num_classes].
        labels: Integer array labels of shape [num_samples].
        top_k: Integer. Number of highest-probability classes to consider.
    Returns:
        float: Top-k accuracy of predictions.
    """
    _, ground_truth = filter_top_k(probabilities, labels, top_k)
    return ground_truth.any(axis=-1).mean()


def bin_predictions_and_accuracies(probabilities, ground_truth, bins=10):
    """Computes histograms of probabilities into bins.

    Args:
        probabilities: A numpy vector of N probabilities assigned to 
            each prediction
        ground_truth: A numpy vector of N ground truth labels in 
            {0,1, True, False}
        bins: Number of equal width bins to bin predictions into in [0, 1], 
            or an array representing bin edges.

    Returns:
        bin_edges: Numpy vector of floats containing the edges of the bins
            (including leftmost and rightmost).
        accuracies: Numpy vector of floats for the average accuracy of the
            predictions in each bin.
        counts: Numpy vector of ints containing the number of examples per bin.
    """
    if isinstance(bins, int):
        num_bins = bins
    else:
        num_bins = bins.size - 1

    probabilities = np.where(probabilities == 0, 1e-8, probabilities)
    counts, bin_edges = np.histogram(probabilities, bins=bins, range=[0., 1.])
    indices = np.digitize(probabilities, bin_edges, right=True)
    accuracies = np.array([np.mean(ground_truth[indices == i])
                           for i in range(1, num_bins + 1)])
    return bin_edges, accuracies, counts


def bin_centers_of_mass(probabilities, bin_edges):
    probabilities = np.where(probabilities == 0, 1e-8, probabilities)
    indices = np.digitize(probabilities, bin_edges, right=True)
    return np.array([np.mean(probabilities[indices == i])
                     for i in range(1, len(bin_edges))])


def expected_calibration_error(probabilities, ground_truth, bins=10):
    """Compute the expected calibration error of a set of preditions in [0, 1].

    Args:
        probabilities: A numpy vector of N probabilities assigned
            to each prediction
        ground_truth: A numpy vector of N ground truth labels in
            {0,1, True, False}
        bins: Number of equal width bins to bin predictions into in [0, 1], or
            an array representing bin edges.
    Returns:
        float: the expected calibration error.  
    """
    bin_edges, accuracies, counts = bin_predictions_and_accuracies(
        probabilities, ground_truth, bins)

    bin_centers = bin_centers_of_mass(probabilities, bin_edges)
    num_examples = np.sum(counts)

    ece = np.sum([(counts[i] / float(num_examples)) * np.sum(
        np.abs(bin_centers[i] - accuracies[i]))
                  for i in range(bin_centers.size) if counts[i] > 0])
    return ece


def expected_calibration_error_multiclass(probabilities, labels, bins=10,
                                          top_k=1):
    """Computes expected calibration error from Guo et al. 2017.

    Args:
        probabilities: Array of probabilities of shape
            [num_samples, num_classes].
        labels: Integer array labels of shape [num_samples].
        bins: Number of equal width bins to bin predictions into in [0, 1], or
            an array representing bin edges.
        top_k: Integer or None. If integer, use the top k predicted
            probabilities in ECE calculation (can be informative for problems
            with many classes and lower top-1 accuracy).
            If None, use all classes.
    Returns:
        float: Expected calibration error.
    """
    top_k_probs, is_correct = get_multiclass_predictions_and_correctness(
        probabilities, labels, top_k)
    top_k_probs = top_k_probs.flatten()
    is_correct = is_correct.flatten()
    return expected_calibration_error(top_k_probs, is_correct, bins)


def compute_accuracies_at_confidences(labels, probs, thresholds):
    """Compute accuracy of samples above each confidence threshold.

    Args:
        labels: Array of integer categorical labels.
        probs: Array of categorical probabilities.
        thresholds: Array of floating point probability thresholds in [0, 1).

    Returns:
        accuracies: Array of accuracies over examples with confidence > T for
            each T in thresholds.
        counts: Count of examples with confidence > T for each T in thresholds.
    """
    assert probs.shape[:-1] == labels.shape

    predict_class = probs.argmax(-1)
    predict_confidence = probs.max(-1)

    shape = (len(thresholds),) + probs.shape[:-2]
    accuracies = np.zeros(shape)
    counts = np.zeros(shape)

    eq = np.equal(predict_class, labels)
    for i, thresh in enumerate(thresholds):
        mask = predict_confidence >= thresh
        counts[i] = mask.sum(-1)
        accuracies[i] = np.ma.masked_array(eq, mask=~mask).mean(-1)

    return accuracies, counts


def compute_calibration(y, p_mean, num_bins=10):
    """Compute the calibration.
        References:
        https://arxiv.org/abs/1706.04599
        https://arxiv.org/abs/1807.00263

    Args:
        y: numpy array, shape [num_classes], the true labels.
        p_mean: numpy array, size (?, num_classes)
                containing the mean output predicted probabilities
        num_bins: number of bins

    Returns:
        cal: a dictionary
            {reliability_diag: realibility diagram
            ece: Expected Calibration Error
            mce: Maximum Calibration Error
            }
    """
    # Compute for every test sample x, the predicted class.
    class_pred = np.argmax(p_mean, axis=1)

    # Convert labels to one-hot encoding
    y = to_one_hot(y)

    # Compute the confidence (probability) associated with it.
    conf = np.max(p_mean, axis=1)

    # Convert y from one-hot encoding to the number of the class
    y = np.argmax(y, axis=1)

    # Storage
    acc_tab = np.zeros(num_bins)  # empirical (true) confidence
    mean_conf = np.zeros(num_bins)  # predicted confidence
    nb_items_bin = np.zeros(num_bins)  # number of items in the bins
    tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins

    for i in np.arange(num_bins):  # iterate over the bins
        # Select the items where the predicted max probability falls in the bin
        sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
        nb_items_bin[i] = np.sum(sec)  # Number of items in the bin

        # Select the predicted classes, and the true classes
        class_pred_sec, y_sec = class_pred[sec], y[sec]

        # Average of the predicted max probabilities
        mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan

        # Compute the empirical confidence
        acc_tab[i] = np.mean(
            class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

    # Cleaning
    mean_conf = mean_conf[nb_items_bin > 0]
    acc_tab = acc_tab[nb_items_bin > 0]
    nb_items_bin = nb_items_bin[nb_items_bin > 0]

    # Expected Calibration Error
    ece = np.average(
        np.absolute(mean_conf - acc_tab),
        weights=nb_items_bin.astype(np.float) / np.sum(nb_items_bin))

    # Maximum Calibration Error
    mce = np.max(np.absolute(mean_conf - acc_tab))

    # Reliability diagram
    reliability_diag = (mean_conf, acc_tab)

    # Saving
    cal = {'reliability_diag': reliability_diag,
           'ece': ece,
           'mce': mce}

    return cal
