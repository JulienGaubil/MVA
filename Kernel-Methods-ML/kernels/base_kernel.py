"""
Implements abstract kernel
"""
from abc import ABC, abstractmethod
import numpy as np
import time
from tqdm import tqdm

import multiprocessing
import joblib as jl

class AbstractKernel(ABC):

    def __init__(self, K_tr=None, K_te=None, X_tr=None, X_te=None, verbose=True):
        '''
            Base class for kernels
        '''
        self.cur_K = None
        self.X_tr = X_tr
        self.X_te = X_te
        self.K_tr = K_tr
        self.K_te = K_te
        self.verbose = verbose
        
    @property
    def X_tr(self):
        if self._X_tr is None:
            raise ValueError("This kernel doesn't have train data")
        return self._X_tr

    @X_tr.setter
    def X_tr(self, X):
        '''
            Parameters
            ----------
            X: list(object).
                Training data.
        '''
        self._X_tr = X
        self.K_tr = None

    @property
    def X_te(self):
        if self._X_te is None:
            raise ValueError("This kernel doesn't have test data")
        return self._X_te

    @X_te.setter
    def X_te(self, X):
        '''
            Parameters
            ----------
            X: list(object).
                Testing data.
        '''
        self._X_te = X
        self.K_te = None
    
    @property
    def K_tr(self):
        if self._K_tr is None:
            self._K_tr = self.compute_K(self.X_tr)
        return self._K_tr

    @K_tr.setter
    def K_tr(self, K):
        '''
            Parameters
            ----------
            K: np.array (shape=(n,n)).
                Gram matrix of the kernel for the training data.
        '''
        self._K_tr = K

    @property
    def K_te(self):
        if self._K_te is None:
            self._K_te = self.compute_K(self.X_te, self.X_tr)
        return self._K_te

    @K_te.setter
    def K_te(self, K):
        """
            Parameters
            ----------
            K_te: np.array (shape=(n_test,n_train)).
                Matrix Kx=(X_te[i], X_tr[j]) of the kernel for the testing data.
        """
        self._K_te = K

    @abstractmethod
    def evaluate(self, x, y):
        pass

    def __call__(self,x,y):
        return self.evaluate(x,y)
    
    def fill_train_line(self, X, i):
        line = np.zeros((i + 1,))
        for j in range(i + 1):
            line[j] = self.evaluate(X[i], X[j])
        return line
    
    def fill_train_line_cross(self, X1, X2, i):
        line = np.zeros((len(X2),))
        for j in range(len(X2)):
            line[j] = self.evaluate(X1[i], X2[j])
        return line

    def center_K(self, K):
        n = K.shape[0]
        U = np.ones(K.shape) / n
        K = (np.eye(n) - U).dot(K).dot(np.eye(n) - U)
        return K
    
    def compute_K(self, X1, X2=None, multi=True):
        """
            Evaluates the the kernel on two sets of graphs X1, X2.
            Parameters
            ----------
            X1: list(object) len n.
                Data.
            X2: list(object) len m.
                Optional. Data if applicable, else computes the Gram matrix.
            Returns
            ----------
            Kx: np.array (shape=(n,m)).
                Matrix of the kernel K(X1[i], X2[j])
        """
        if self.verbose:
            start = time.time()
            print('Starts computing kernel')     
        n = len(X1)
        if X2 is None: #special case, saves half of the computation for symmetric K
            K = np.zeros((n,n), dtype=np.float32)

            if multi: #computes with multithreading
                results_lines = jl.Parallel(n_jobs=multiprocessing.cpu_count())(jl.delayed(self.fill_train_line)(X1, i) for i in tqdm(range(n - 1, -1, -1)))
                for i in range(n):
                    K[:i + 1, i] = results_lines[n - 1 - i]
            
            else: #computes with monothreading
                for i in tqdm(range(n)):
                    for j in range(i,n):
                        K[i,j] = self.evaluate(X1[i],X1[j])
        
            K = K + K.T - np.diag(np.diag(K))
        
        else:
            m = len(X2)
            K = np.zeros([n,m], dtype=np.float32)

            if multi:
                results_lines = jl.Parallel(n_jobs=multiprocessing.cpu_count())(jl.delayed(self.fill_train_line_cross)(X1, X2, i) for i in tqdm(range(n - 1, -1, -1)))
                for i in range(n):
                    K[i, :] = results_lines[n - 1 - i]

            else: 
                for i in tqdm(range(n)):
                    for j in range(m):
                        K[i,j] = self.evaluate(X1[i],X2[j])

        if self.verbose:    
            end = time.time()
            print('Kernel computation finished in {0:.2f}'.format(end-start))    

        return K



    




        

