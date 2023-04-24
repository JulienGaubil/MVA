import numpy as np
from models.base_model import AbstractKernelAlgorithm
from cvxopt import matrix, solvers
import time
 

class CSVM(AbstractKernelAlgorithm):

    def __init__(self, kernel, C=0.01, tol=10**-8, verbose=True):
        super().__init__(kernel, verbose)
        self.C = C
        self.tol = tol
        self.support_vectors = None
        self.active_idx = None

    def train(self, y, X=None):
        """
            Solves the dual SVM problem.
            Parameters
            ----------
            y: np.array (shape=(n,))
                Ground truth labels {-1,1}.
            X: list(object).
                Optional. Training data.
        """
        
        K = np.copy(self.K_tr) if X is None else self.kernel.compute_K(X)
        if X is None:
            X = self.X_tr.copy()
        
        n = K.shape[0]

        A = y.reshape(1, -1).astype('float')
        b = np.zeros(1, dtype=float)
        G = np.vstack((-np.eye(n), np.eye(n)), dtype=float)
        h = np.hstack((np.zeros(n), self.C * np.ones(n))).astype('float')
        P = np.einsum('i,j,ij->ij', y, y, K).astype('float')
        q = -np.ones(n, dtype=float)
        
        solvers.options['show_progress'] = self.verbose
        res = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))['x']
        self._alpha_full = np.array(res).flatten()

        #gets support vectors
        active_supports = (self._alpha_full > self.tol)
        margin_idx = (self.tol < self.C - self._alpha_full) * active_supports
        self.support_vectors = [X[idx] for idx in margin_idx]

        #gets active constraints and separative function
        self.active_idx = np.where(active_supports)[0]
        self._alpha = self._alpha_full[self.active_idx]*y[self.active_idx]
        f = K[self.active_idx,:][:,self.active_idx].dot(self._alpha)
        self._b = (y[self.active_idx]-f).mean()

        if self.verbose:
            print(f'[C-SVM]: Number of active supports: {len(self.active_idx)}/{n}')

    def predict_values(self, X=None, Kx=None, train=False, test=False):
        """
            Regresses values from the train/test Gram matrix of the kernel or the input data X if specified.
            Parameters
            ----------
            X: list(object).
                Optional. Training data.
            train: bool.
                Optional. Whether to use training data as input or not.
            test: bool.
                Optional. Whether to use test data as input or not.  
            Returns
            ----------
            preds: np.array (shape=(n,)).
                Values regressed for each sample.
        """
        if self.verbose:
            print('[C-SVM]: Starts prediction')
            start = time.time()
        
        if Kx is None:
            if train:
                Kx=self.K_tr[self.active_idx,:][:,self.active_idx]
            elif test:
                Kx=self.K_te
            else:
                if X is None:
                    raise ValueError("No input data provided to compute Kx.")
                Kx = self.kernel.compute_K(X,self.X_tr)
        preds = Kx.dot(self.alpha) + self.b

        end = time.time()
        if self.verbose:
            print('[C-SVM]: Predictions computed in {0:.2f}'.format(end-start))
        return preds.reshape(-1)

    
    @property
    def X_tr(self):
        if self.active_idx is None:
            return self.kernel.X_tr
        else:
            return [self.kernel.X_tr[idx] for idx in self.active_idx]
    
    @property
    def K_te(self):
        if self.active_idx is None:
            return self.kernel.K_te
        else:
            return self.kernel.K_te[:,self.active_idx]






