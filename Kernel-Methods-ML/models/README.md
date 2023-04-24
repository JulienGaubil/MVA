# Models documentation

The kernel learning algorithms implemented are:

- Kernel C-SVM
- Kernel Ridge Regression
- Kernel Logistic Regression

All the algorithms are subclasses of the abstract class `AbstractKernelAlgorithm` that provides a framework. It contains methods `train` and `predict_values` that enable to train the classifier given a kernel and to regress values on a given dataset. 

## KRR
The `train` method obtains optimal solution for KRR problem: $$\alpha = (K+C~I_d)^{-1}~y$$

The only hyperparameter is:

```
- C                     Regularization parameter in the formulation of KRR
```

## KLR
The `train` method implements Newton's method as a problem of Weighted Kernel Ridge Regression.

The hyperparameters are:

```
- C                     Regularization parameter in the formulation of KLR
- tol                   Tolerance in Newton's method
- n_iter_max            Maximum number of iterations in Newton's method
```


## C-SVM
The `train` method solves the dual SVM problem with the quadratic program solver of CVXOpt. Only active supports are used for prediction. Some kernels (Geometric and N-th order random walk) may not be compatible with this algorithm for non-singular matrices issues.

The hyperparameters are:

```
- C                     Regularization parameter in the penalized formulation of the C-SVM
- tol                   Tolerance for the active supports
```