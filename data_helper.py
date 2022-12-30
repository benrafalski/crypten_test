import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import crypten
import time
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from statistics import mean


def _get_norm_mnist(dir, reduced=None, binary=False):
    """Downloads and normalizes mnist"""
    mnist_train = datasets.MNIST(dir, download=True, train=True)
    mnist_test = datasets.MNIST(dir, download=True, train=False)

    # compute normalization factors
    data_all = torch.cat([mnist_train.data, mnist_test.data]).float()
    data_mean, data_std = data_all.mean(), data_all.std()
    tensor_mean, tensor_std = data_mean.unsqueeze(0), data_std.unsqueeze(0)

    # normalize
    mnist_train_norm = transforms.functional.normalize(
        mnist_train.data.float(), tensor_mean, tensor_std
    )
    mnist_test_norm = transforms.functional.normalize(
        mnist_test.data.float(), tensor_mean, tensor_std
    )

    # create a reduced dataset if required
    if reduced is not None:
        mnist_norm = (mnist_train_norm[:reduced], mnist_test_norm[:reduced])
        mnist_labels = (mnist_train.targets[:reduced], mnist_test.targets[:reduced])
    else:
        mnist_norm = (mnist_train_norm, mnist_test_norm)
        mnist_labels = (mnist_train.targets, mnist_test.targets)
    return mnist_norm, mnist_labels


def split_features(
    split=0.5, dir="tmp", party1="alice", party2="bob", reduced=None, binary=False
):
    """Splits features between Party 1 and Party 2"""
    mnist_norm, mnist_labels = _get_norm_mnist(dir, reduced, binary)
    mnist_train_norm, mnist_test_norm = mnist_norm
    mnist_train_labels, mnist_test_labels = mnist_labels

    num_features = mnist_train_norm.shape[1]
    split_point = int(split * num_features)

    party1_train = mnist_train_norm[:, :, :split_point]
    party2_train = mnist_train_norm[:, :, split_point:]
    party1_test = mnist_test_norm[:, :, :split_point]
    party2_test = mnist_test_norm[:, :, split_point:]

    torch.save(party1_train, os.path.join(dir, party1 + "_train.pth"))
    torch.save(party2_train, os.path.join(dir, party2 + "_train.pth"))
    torch.save(party1_test, os.path.join(dir, party1 + "_test.pth"))
    torch.save(party2_test, os.path.join(dir, party2 + "_test.pth"))
    torch.save(mnist_train_labels, os.path.join(dir, "train_labels.pth"))
    torch.save(mnist_test_labels, os.path.join(dir, "test_labels.pth"))



def main():
    split_features(
            split=0.72,
            dir="tmp",
            party1="alice",
            party2="bob",
            reduced=100,
            binary=False,
        )

if __name__ == "__main__":
    main()
    













