import json
import pandas as pd
import pickle
import numpy as np
import torch
import os
import argparse

from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def load_uci_data(uci_dir, split_id, name, version="original"):
    """Load a split of a UCI dataset.

    Args:
        data_dir: str, path to the directory containing the UCI datasets.
        split_id: int, the index of the split to be loaded.
        name: str, the name of the dataset.
        version: str, the version of the uci dataset, must be either
            `original` or `gap`.

    Returns:
        x_train: numpy array, the training data points.
        y_train: numpy array, the training labels.
        x_test: numpy array, the test data points.
        y_test: numpy array, the test labels.
    """
    datasets = ["boston", "concrete", "energy", "kin8nm",
                "naval", "power", "protein", "wine", "yacht"]
    if not(name in datasets):
        raise ValueError("Invalid dataset name.")
    assert version in ["original", "gap"]

    uci_dir = os.path.join(uci_dir, name)
    data_file = os.path.join(uci_dir, "data.txt")
    idx_train_file = os.path.join(uci_dir, "{}/index_train_{}.txt").\
        format(version, split_id)
    idx_test_file = os.path.join(uci_dir, "{}/index_test_{}.txt").\
        format(version, split_id)

    data = np.loadtxt(data_file)
    idx_train = np.loadtxt(idx_train_file).astype(np.int32)
    idx_test = np.loadtxt(idx_test_file).astype(np.int32)

    x, y = data[:, :-1], data[:, -1]
    x_train, y_train = x[idx_train, :], y[idx_train]
    x_test, y_test = x[idx_test, :], y[idx_test]

    return x_train, y_train, x_test, y_test


def prepare_device(n_gpu_use):
    """Setup GPU device if available, move model into configured device.

    Args:
        n_gpu_use: number of used GPUs.
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use"
              " is {}, but only {} are available "
              "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))

    return device, list_ids


def to_one_hot(y, num_classes=10):
    """Convert labels to one-hot vectors.

    Args:
        y: numpy array, shape [num_classes], the true labels.

    Returns:
        one_hot: numpy array, size (?, num_classes), 
            array containing the one-hot encoding of the true classes.
    """
    if isinstance(y, torch.Tensor):
        one_hot = torch.zeros((y.shape[0], num_classes), dtype=torch.float32)
        one_hot[torch.arange(y.shape[0]), y] = 1.0
    else:
        one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float)
        one_hot[np.arange(y.shape[0]), y] = 1.0

    return one_hot


def str2bool(v):
    """Convert string to boolean variable"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_pickle(data, file_path):
    """Wrapper for saving data to a pickle file.

    Args:
        data: a dictionary containing the data needs to be saved.
        file_path: string, path to the output file.
    """
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    """Wrapper for loading data from a pickle file.

    Args:
        file_path: string, path to the pickle file.

    Returns:
        A dictionary containing the loaded data.
    """
    with open(file_path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def ensure_dir(dirname):
    """Check whether a given directory was created; if not, create a new one.

    Args:
        dirname: string, path to the directory.
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(file_path):
    """Wrapper for reading a json file.

    Args:
        file_path: string, path to the json file.

    Returns:
        A dictionary containing the loaded data.
    """
    file_path = Path(file_path)
    with file_path.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, file_path):
    """Write data to a json file.

    Args:
        content: a dictionary containing the data needs to be saved.
        file_path: string, path to the output file.
    """
    file_path = Path(file_path)
    with file_path.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def set_seed(seed=99):
    """Set seed for reproducibility purpose."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_all_data(data_loader):
    """Get all data from a data loader."""
    x, y = [], []
    for x_batch, y_batch in data_loader:
        x.append(x_batch)
        y.append(y_batch.reshape([-1, 1]))

    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0).reshape([-1])

    return x, y


def split_train_val(x_train, y_train, splitting_ratio=0.2):
    """Split the data into training and validation set.
    """
    num_samples = x_train.shape[0]
    num_train_samples = int(num_samples * (1 - splitting_ratio))

    indices = np.random.permutation(num_samples)
    train_idx = indices[:num_train_samples]

    val_idx = indices[num_train_samples:]
    x_val, y_val = x_train.copy()[val_idx, :], y_train.copy()[val_idx]
    x_train, y_train = x_train.copy()[train_idx, :], y_train.copy()[train_idx]

    return x_train, y_train, x_val, y_val
