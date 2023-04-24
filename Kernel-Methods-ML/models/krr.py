import numpy as np
from models.base_model import AbstractKernelAlgorithm


class KRR(AbstractKernelAlgorithm):

    def __init__(self, kernel, C=0.001, verbose=True):
        super().__init__(kernel, verbose)   
        self.C = C
       

    def train(self, y, X=None):
        """
            Solves KRR problem.
            Parameters
            ----------
            y: np.array (shape=(n,))
                Ground truth labels {-1,1}.
            X: list(object).
                Optional. Training data.
        """
        K = self.K_tr if X is None else self.kernel.compute_K(X)
        dim = K.shape[0]

        self._alpha = np.linalg.inv(K+self.C*dim*np.eye(dim))@y

    

