import numpy as np
import pandas as pd
from argparse import ArgumentParser
import os, os.path

from utils.tools import preprocessing_nodes, process_args, load_kernel
from utils.kfold_validation import kfold_validation

from models.svm import CSVM
from models.krr import KRR
from models.klr import KLR

from kernels.geometric_walk_kernel import GeometricWalkKernel
from kernels.nth_order_walk_kernel import NthOrderWalkKernel
from kernels.wl import WL


parser = ArgumentParser()
parser.add_argument(
    '--path_data',
    type=str,
    default='data/',
    help='Dataset folder path')
parser.add_argument(
    '--experiment_all',
    default=False,
    action='store_true',
    help='Whether to benchmark all kernels and classifiers')
parser.add_argument(
    '--kernel',
    type=str,
    default="wl",
    help="Kernel to use ('wl', 'geometric', 'nth')")
parser.add_argument(
    '--h',
    type=int,
    default=3,
    help='Number of iterations in the WL algorithm')
parser.add_argument(
    '--no_edges',
    default=True,
    action='store_false',
    help='Use edges labels in WL or only nodes')
parser.add_argument(
    '--lbd',
    type=float,
    default=0.1,
    help='Geometric coefficient in the Geometric kernel')
parser.add_argument(
    '--n',
    type=int,
    default=4,
    help='Order of the walk in the N-th order random walk kernel')
parser.add_argument(
    '--model',
    type=str,
    default="svm",
    help="Classifier to use ('svm', 'krr', 'klr')")
parser.add_argument(
    '--C',
    type=float,
    help='Regularization parameter for the classifier')
parser.add_argument(
    '--tol',
    type=float,
    default=10**-8,
    help='Tolerance for the classifier')
parser.add_argument(
    '--n_iter_max',
    type=int,
    default=10,
    help='Maximal number of iterations in KLR algorithm')
parser.add_argument(
    '--load_kernel',
    default=False,
    action='store_true',
    help='Whether to use a pre-computed kernel for train and test')
parser.add_argument(
    '--k_folds',
    type=int,
    default=5,
    help='Number of folds')


args = parser.parse_args()

#loads graphs and preprocess nodes labels
train_graphs = pd.read_pickle(os.path.join(args.path_data, 'training_data.pkl'))
train_graphs = np.array(train_graphs, dtype=object)
train_graphs = preprocessing_nodes(train_graphs)

train_labels = pd.read_pickle(os.path.join(args.path_data, 'training_labels.pkl'))
train_labels = 2*train_labels-1
kernels = {'wl':WL, 'geometric':GeometricWalkKernel, 'nth':NthOrderWalkKernel }
classifiers = {'svm':CSVM, 'krr':KRR, 'klr':KLR}

if args.experiment_all:

    errors = [('geometric', 'svm')]

    for ker in kernels.keys():
        k = kernels[ker](X_tr=train_graphs, verbose=False)
        K = load_kernel(ker, split='train') if args.load_kernel else k.compute_K(train_graphs)
        for clf_ in classifiers.keys():
            if not (ker, clf_) in errors:
                clf = classifiers[clf_](kernel=k, verbose=False)
                print('-----------------------------------------------------------------')
                print(f'Benchmarking kernel {ker} with classifier {clf_} ')
                kfold_validation(clf,train_labels,kfolds=args.k_folds, X=train_graphs, K=K)
                print('')
                print('')

else:
    #loads kernel and classifier
    kernels = {'wl':WL, 'geometric':GeometricWalkKernel, 'nth':NthOrderWalkKernel }
    classifiers = {'svm':CSVM, 'krr':KRR, 'klr':KLR}
    args_kernel, args_model = process_args(args)
    k = kernels[args.kernel](X_tr=train_graphs, verbose=False, **args_kernel)
    K = load_kernel(args.kernel, split='train') if args.load_kernel else k.compute_K(train_graphs)
    clf = classifiers[args.model](kernel=k, verbose=False, **args_model)

    #trains classifier and regresses values on test set
    print('-----------------------------------------------------------------')
    print(f'Benchmarking kernel {args.kernel} with classifier {args.model} ')
    kfold_validation(clf,train_labels,kfolds=args.k_folds, X=train_graphs, K=K)








