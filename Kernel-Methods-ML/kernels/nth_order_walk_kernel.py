import numpy as np
import networkx as nx
import torch
import time
from tqdm import tqdm
from itertools import product

from scipy import sparse

from kernels.base_kernel import AbstractKernel
from utils.tools import bin_coeffs_vec, compute_sum_pows



class NthOrderWalkKernel(AbstractKernel):

    def __init__(self, K_tr=None, K_te=None, X_tr=None, X_te=None, n=4, verbose=True):
        super().__init__(K_tr, K_te, X_tr, X_te, verbose)
        self.n=n
        self.evaluate_fn = self.evaluate_torch if torch.cuda.is_available() else self.evaluate_np
    
    def evaluate_np(self, A):
        return (A**self.n).sum() 
    
    def evaluate_torch(self,A):
        return torch.sum(torch.linalg.matrix_power(A, self.n)) 
    
    def evaluate(self, x, y):      
        A_prod = self.compute_adj_prod(x,y)   
        return self.evaluate_fn(A_prod)    

    def prepare_kernels(self,X):
        """
          Returns adjency matrices filtered by node labels for a list of graphs X.
          Parameters
          ----------
          X : list(networkX graph) len n
              List of input graphs.
          Returns
          -------
          adj_filtered : list(dict) len n
              For each graph, returns a dict of its adjacency matrices filtered by each pair of node labels.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #gets all graphs nodes labels
        graphs = list()
        for graph in X:            
            adj = nx.to_numpy_array(graph)
            labels = np.array([graph._node[k]['labels'][0] for k in range(adj.shape[0])]) #labels for each node
            graphs.append((adj, labels))

        #creates adjency matrix filtered by nodes labels
        adj_filtered = list()
        for adj, labels in graphs:
            adj_ = dict()
            labels_ = set(labels)

            for t in product(labels_, labels_):  #runs through all pairs of labels
                selector = np.matmul(np.expand_dims(labels == t[0], axis=1), np.expand_dims(labels == t[1], axis=0))
                adj_t = adj * selector  #adjacency matrix for nodes (i,j) connected so that label[i]=t[0], label[j]=t[1]
                
                adj_[t] = torch.tensor(adj_t, dtype=torch.float32, device=device) if torch.cuda.is_available() else sparse.csr_matrix(adj_t)

            adj_filtered.append(adj_)

        return adj_filtered
    
    def compute_adj_prod(self,x,y):
        """
        Computes the adjacency matrix of the product graph of two labelled graphs.
        Parameters
        ----------
        x, y : dicts
            Dicts of filtered adjacency matrices wrt labels of two graphs.
        Returns
        -------
        adj : np.array (shape=(n*m,n*m)).
            Adjacency matrix of the graph product of graphs X, Y.

        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xs = list(x.values())[0].shape[0] # nb of nodes graph X
        ys = list(y.values())[0].shape[0] # nb of nodes graph Y
        m = xs * ys  #size kronecker product matrix Wx

        common_edge_labels = set(x.keys()) & (set(y.keys())) #common edge labels between X and Y

        m = xs * ys  #size kronecker product matrix Wx
        adj_ = [(x[k], y[k]) for k in common_edge_labels]  # for bloc computation with Kronecker product matrix between adjency matrix of graphs X and Y
        
        if len(common_edge_labels):
            adj=None
        
            for adjx, adjy in adj_:
                k = torch.kron(adjx, adjy) if torch.cuda.is_available() else sparse.kron(adjx, adjy)
                if adj is not None:
                    adj += k
                else:
                    adj = k
            return adj
        else:
            return np.zeros((m,m))
    
    def compute_K(self, X1, X2=None):
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
        multi = not torch.cuda.is_available()
        features1 = self.prepare_kernels(X1)
        if X2 is None:
            return super().compute_K(features1, multi=multi)
        else:
            features2 = self.prepare_kernels(X2)
            return super().compute_K(features1, features2, multi=multi)