import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.sparse import csr_matrix

from base import OnlineSolver


class RDASolver(OnlineSolver):
    def __init__(self, w_dim, psi, aux, betas):
        """Implement the Regularized Dual Avering method [Xiao 2009] (cf. Algorithm 1)

        Args:
            w_dim (int): the dimension of the solution vector
            psi (func): penalization function (input: vector of shape (w_dim, ), output: float)
            aux (func): auxiliary function defined in the paper
            betas (iterator): non-negative non-decreasing series of float numbers
        """
        self._t = 1
        self._gBar = csr_matrix((1, w_dim))
        self.psi = psi  # penalization
        self.aux = aux  # auxiliary function
        self.betas = betas
        w, _, _ = fmin_l_bfgs_b(self.aux, np.zeros(w_dim), approx_grad=True)
        self.w = csr_matrix(w)

    def iterate(self, ft, gt=None):
        if gt is None:
            raise NotImplementedError("Should compute a subgradient of the loss function at time t")

        # We compute the average of the past gradients
        self._gBar = (self._t - 1) / self._t * self._gBar + 1 / self._t * gt

        # We can then define the objective function to minimize
        def obj(w):
            return np.dot(self._gBar, w) + self.psi(w) + next(self.betas) / self._t * self.aux(w)

        w_, _, _ = fmin_l_bfgs_b(obj, self.w, approx_grad=True)
        self.w = csr_matrix(w_)
        self._t += 1
        return self.w

    def status(self):
        data = {"t": self._t, "w": self.w, "gBar": self._gBar}
        return data
