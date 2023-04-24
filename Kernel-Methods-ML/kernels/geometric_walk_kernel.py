import numpy as np
import networkx as nx
from kernels.base_kernel import AbstractKernel
from scipy.sparse.linalg import cg, LinearOperator
from itertools import product


class GeometricWalkKernel(AbstractKernel):

    def __init__(self, K_tr=None, K_te=None, X_tr=None, X_te=None, lbd=0.1, verbose=True):
        super().__init__(K_tr, K_te, X_tr, X_te, verbose)
        self.lbd=lbd
    
    def prepare_kernels(self,X):
        """
            Returns adjency matrixes filtered by node labels for Kronecker product matrix computation between features.
            Parameters
            ----------
            X : list(networkX graph) len n
                List of input graphs.
            Returns
            -------
            features : list(dict) len n
                For each graph, returns a dict of its adjacency matrix filtered by each pair of node labels.
        """
       #gets all graphs nodes labels
        graphs = list()
        for graph in X:            
            adj = nx.to_numpy_array(graph)
            labels = np.array([graph._node[k]['labels'][0] for k in range(adj.shape[0])]) #labels for each node
            graphs.append((adj, labels))

        #creates adjency matrix filtered by nodes labels
        features = list()
        for adj, labels in graphs:
            adj_ = dict()
            labels_ = set(labels)

            for t in product(labels_, labels_):  #runs through all pairs of labels
                selector = np.matmul(np.expand_dims(labels == t[0], axis=1), np.expand_dims(labels == t[1], axis=0))
                adj_t = adj * selector  #adjacency matrix for nodes (i,j) connected so that label[i]=t[0], label[j]=t[1]
                adj_[t] = adj_t

            features.append(adj_)

        return features

    def evaluate(self,X,Y):
        """Calculate the geometric random walk kernel by Conjugate Gradient method described in
        https://www.jmlr.org/papers/volume11/vishwanathan10a/vishwanathan10a.pdf section .4.2
        Parameters
        ----------
        X, Y : dicts
            Dicts of adjacency matrices wrt labels.
        Returns
        -------
        k : float
            Kernel evaluation in (X,Y).

        """
        xs = list(X.values())[0].shape[0] # nb of nodes graph X
        ys = list(Y.values())[0].shape[0] # nb of nodes graph Y
        m = xs * ys  #size kronecker product matrix Wx

        common_edge_labels = set(X.keys()) & (set(Y.keys())) #common edge labels between X and Y

        m = xs * ys  #size kronecker product matrix Wx

        Wx = [(X[k], Y[k]) for k in common_edge_labels]  # for bloc computation with Kronecker product matrix between adjency matrix of graphs X and Y

        #defines linear operator for conjugate gradient
        def prod(x):
            '''Defines product (Id - lambda W_x) x'''
            y = np.zeros(m)
            xm = x.reshape((xs, ys))
            for Ax, Ay in Wx:
                y += np.reshape(np.linalg.multi_dot((Ax, xm, Ay)), (m,)) #bloc compute of Wx.dot(x)
            return x - self.lbd * y

        # solves (I_d - lmbd W_x)*x=p_x with conjugate gradient
        A = LinearOperator((m, m), matvec=prod)
        px = np.ones(m)
        x_sol, _ = cg(A, px, tol=1.0e-6, maxiter=20, atol='legacy')
      
        return np.sum(x_sol)
    
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
        features1 = self.prepare_kernels(X1)
        if X2 is None:
            return super().compute_K(features1, multi=multi)
        else:
            features2 = self.prepare_kernels(X2)
            return super().compute_K(features1, features2, multi)