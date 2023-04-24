import numpy as np
from utils.tools import sigmoid
from models.base_model import AbstractKernelAlgorithm
from tqdm import tqdm

class KLR(AbstractKernelAlgorithm):
    """Class for Kernel Logistic Regression."""

    def __init__(self, kernel, C=0.001, n_iter_max=10, tol=10**-8, verbose=True):
        super().__init__(kernel, verbose)     
        self.C = C
        self.tol = tol
        self.n_iter_max = n_iter_max

    def train(self, y, X=None):
        """
        Newton method for Kernel Logistic Regression.
        Parameters
        ----------
        y: np.array (shape=(n,))
            Ground truth labels {-1,1}.
        X: list(object).
            Optional. Training data.
        """
        K = self.K_tr if X is None else self.kernel.compute_K(X)
        n = K.shape[0]

        alpha_prec = np.zeros(n)+np.inf
        alpha= np.zeros(n)

        for k in range(self.n_iter_max):
            M = K.dot(alpha)
            sig= sigmoid(M * y)
            hess = sigmoid(-M * y) * sig
            Z = M + y / np.maximum(sig, self.tol)
            alpha_prec = alpha

            #Solves Weighted Kernel Ridge Regression
            hess_half = np.diag(np.sqrt(hess))
            inv = np.linalg.inv( hess_half.dot(K.dot(hess_half)) + n * self.C * np.eye(n) )
            alpha = hess_half.dot(inv.dot(hess_half.dot(Z)))

            if not (np.abs(alpha- alpha_prec) > self.tol).any():
                break
            elif k==self.n_iter_max-1 and self.verbose:
                print(f'[KLR]: KLR algorithm exited in {self.n_iter_max} iterations without converging')

        self._alpha = alpha