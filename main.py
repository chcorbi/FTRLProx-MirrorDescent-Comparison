import argparse

import numpy as np
from scipy import stats
from sklearn.datasets import load_svmlight_file

from base import OnlineClassifier
from solvers import *


def loading_dataset(filename):
    data = load_svmlight_file(filename)
    return data[0], data[1]


def nnz_fraction(X):
    """Compute sparsity fraction in a vector"""
    return X.nnz / X.shape[1]


def sigmoid(x):
    """Sigmoid function"""
    return 1. / (1 + np.exp(-x))


def main(datafile):
    X, y = loading_dataset(datafile)
    print('y:', stats.describe(y))
    # TODO: define an OnlineClassifier instance and train it over the dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help="One input file")
    args = parser.parse_args()
    main(args.input_file)
