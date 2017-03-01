import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator

from utils import *


class SqrtIterator():
    def __init__(self, gamma=1.0):
        self._t = 1
        self.gamma = gamma

    def __iter__(self):
        return self

    def __next__(self):
        sq = np.sqrt(self._t)
        self._t += 1
        return self.gamma * sq


class RDASolver(BaseEstimator):
    def __init__(self, lbda, gamma):
        """Implement the Regularized Dual Avering method [Xiao 2009] (cf. Algorithm 1)

        Args:
            w_dim (int): the dimension of the solution vector
            psi (func): penalization function (input: vector of shape (w_dim, ), output: float)
            aux (func): auxiliary function defined in the paper
            betas (iterator): non-negative non-decreasing series of float numbers
        """
        self.lbda = lbda
        self.gamma = gamma
        self._t = 1
        self._gBar = None
        self.betas = SqrtIterator(gamma)
        self.w = None
        self.losses = []
        self.log_likelihood = 0

    def iterate(self, X, y, g):
        # We compute the average of the past gradients
        self._gBar = (self._t - 1) / self._t * self._gBar + 1 / self._t * g

        # We can then define the objective function to minimize
        beta_t = next(self.betas)

        nnz_X = X.nonzero()[1]
        if nnz_X.size == 0:
            raise "Error at ligne %d " % (self._t + 1)

        # Update weight
        for i in nnz_X:
            if self._gBar[0, i] <= self.lbda:
                self.w[0, i] = 0
            else:
                sign = 1. if self._gBar[0, i] >= 0 else -1.
                self.w[0, i] = - beta_t / self.gamma * (self._gBar[0, i] - self.lbda * sign)

        # Compute probability and losses
        wtx = self.w.dot(X.T)[0, 0]
        p = sigmoid(wtx)
        self.log_likelihood += log_loss(y, p)
        self.losses.append(self.log_likelihood)

        self._t += 1
        return self.w, p

    def status(self):
        data = {"t": self._t, "w": self.w, "gBar": self._gBar}
        return data

    def train(self, X, y):
        w_dim = X.shape[1]
        self.w = csr_matrix((1, w_dim))
        self._gBar = np.zeros((1, w_dim))
        y_proba = []

        for t in range(X.shape[0]):
            g = X[t] / (1 + np.exp(-y[t] * self.w.dot(X[t].T)[0, 0]))
            w, p = self.iterate(X[t], y[t], g)
            y_proba.append(p)
            if t % int(X.shape[0] / 10) == 0:
                print("Training Samples: %d |\t Loss: %s" % (t, self.log_likelihood))

        return self.w, y_proba

    def predict_proba(self, X):
        return self.w.dot(X.T)

    def predict(self, X):
        probas = self.predict_proba(X)
        return sigmoid(probas) >= 0.5
