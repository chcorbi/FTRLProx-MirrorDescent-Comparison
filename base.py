import numpy as np


def sign(x):
    return 1. if x >= 0 else -1.


class OnlineSolver:
    def __init__(self, *args, **kwargs):
        pass

    def iterate(self, f, *args, **kwargs):
        pass


class OnlineClassifier:
    def __init__(self, solver):
        self.solver = solver
        self.w = None

    def fit(self, X, y):
        T = X.shape[0]
        for t in range(T):
            def loss(w):
                return np.log(1 + np.exp(- y[t] * np.dot(w, X[t])))
            self.w = self.solver.solve(loss)

    def predict(self, X):
        return np.vectorize(sign)(np.dot(self.w, X))
