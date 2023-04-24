import numpy as np
import pandas as pd
from argparse import ArgumentParser
import os, os.path

from utils.tools import preprocessing_nodes, process_args, load_kernel

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
    default=0.01,
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
    '--test',
    default=False,
    action='store_true',
    help='Predicts values on the test set and save predictions')
parser.add_argument(
    '--load_kernel',
    default=False,
    action='store_true',
    help='Whether to use a pre-computed kernel for train and test')

args = parser.parse_args()

#loads graphs and preprocess nodes labels
train_graphs = pd.read_pickle(os.path.join(args.path_data, 'training_data.pkl'))
train_graphs = np.array(train_graphs, dtype=object)
train_graphs = preprocessing_nodes(train_graphs)

test_graphs = pd.read_pickle(os.path.join(args.path_data, 'test_data.pkl'))
test_graphs = np.array(test_graphs, dtype=object)
test_graphs = preprocessing_nodes(test_graphs)

train_labels = pd.read_pickle(os.path.join(args.path_data, 'training_labels.pkl'))
train_labels = 2*train_labels-1

#loads kernel and classifier
kernels = {'wl':WL, 'geometric':GeometricWalkKernel, 'nth':NthOrderWalkKernel }
classifiers = {'svm':CSVM, 'krr':KRR, 'klr':KLR}
args_kernel, args_model = process_args(args)
k = kernels[args.kernel](**args_kernel)
clf = classifiers[args.model](kernel=k, **args_model)

#trains classifier
k.X_tr = train_graphs
k.K_tr = load_kernel(args.kernel, split='train') if args.load_kernel else k.compute_K(train_graphs)
clf.train(train_labels)

#regresses values on test set
if args.test:
    k.X_te = test_graphs
    k.K_te = load_kernel(args.kernel, split='test') if args.load_kernel else k.compute_K(test_graphs,train_graphs)
    y_pred = clf.predict_values(test=True)
    Yte = {'Predicted': y_pred}
    dt = pd.DataFrame(Yte)
    dt.index +=1
    dt.to_csv('test_pred.csv', index_label='Id')








