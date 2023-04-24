from abc import ABC, abstractmethod
import time
import numpy as np
from utils.metrics import compute_auc_precision_recall

class AbstractKernelAlgorithm(ABC):

    def __init__(self, kernel, verbose=True):
        self.kernel = kernel
        self._alpha = None
        self._b = None
        self.verbose = verbose

    @property
    def alpha(self):
        if self._alpha is None:
            raise ValueError("Model has not yet been trained")
        return self._alpha.reshape(self._alpha.size, 1)
    
    @property
    def b(self):
        if self._b is None:
            return 0
        return self._b
    
    @property
    def X_tr(self):
        return self.kernel.X_tr
    
    @property
    def X_te(self):
        return self.kernel.X_te
    
    @property
    def K_tr(self):
        return self.kernel.K_tr
    
    @property
    def K_te(self):
        return self.kernel.K_te

    @abstractmethod
    def train(self, y, X=None):
        pass

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
            print('Starts prediction')
            start = time.time()

        if Kx is None:
            if train:
                Kx=self.K_tr
            elif test:
                Kx=self.K_te
            else:
                if X is None:
                    raise ValueError("No input data provided to compute Kx.")
                Kx = self.kernel.compute_K(X,self.X_tr)
                
        preds = Kx.dot(self.alpha) + self.b

        if self.verbose:
            end = time.time()
            print('Predictions computed in {0:.2f}'.format(end-start))

        return preds.reshape(-1)
    
    def predict_classes(self, X=None, train=False, test=False):
        """
            Predicts classes from the train/test Gram matrix of the kernel or the input data X if specified.
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
            classes: np.array (shape=(n,)).
                Binary classes {-1,1} predicted for each sample.
        """
        classes = np.sign(self.predict_values(X, train, test))
        return classes
    
    def evaluate(self,labels, X=None, train=False, test=False):
        """
            Predicts classes from the train/test Gram matrix of the kernel or the input data X if specified.
            Parameters
            ----------
            labels: np.array (shape=(n,)).
                Binary ground-truth labels {-1,1} for the samples.
            X: list(object).
                Optional. Training data.
            train: bool.
                Optional. Whether to use training data as input or not.
            test: bool.
                Optional. Whether to use test data as input or not.  
            Returns
            ----------
            auc: float.
                Area Under the Curve precision-recall for the predicted values.
        """
        preds = self.predict_values(X,train,test)
        auc = compute_auc_precision_recall(labels,preds)
        return auc



