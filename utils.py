import numpy as np


def nnz_fraction(w):
    """Compute sparsity fraction in a dictionnary"""
    a = np.array([i for i in w.values()])
    return np.sum(a != 0) / a.shape[0]


def sigmoid(x):
    """Sigmoid function"""
    return 1. / (1 + np.exp(-x))


def log_loss(y, p):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -np.log(p) if y == 1 else -np.log(1. - p)
