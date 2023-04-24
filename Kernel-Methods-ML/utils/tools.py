import numpy as np
import os, os.path
import torch
import math
import networkx as nx


def save_kernel(K,dirpath,filename):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    np.save(os.path.join(dirpath,filename), K)

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def bin_coeffs_vec(n):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return torch.tensor([math.comb(n,k) for k in range(n+1)], dtype=torch.float32, device=device)

def compute_sum_pows(A, n):
    """Computes sum(A^k) for k in [0,n]"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = torch.empty(size=(n+1,), dtype=torch.float32, device=device)
    for k in range(n+1):
      out[k] = torch.sum(torch.linalg.matrix_power(A, k))
    return out

def preprocessing_nodes(graphs):
    """Converts list node labels in string node labels"""
    for graph in graphs:
        nx.set_node_attributes(graph, {k: str(v[0]) for k, v in graph.nodes(data='labels')}, 'labels')
    return graphs

def process_args(args):

    #processes kernel arguments
    if args.kernel=="wl":
        args_kernel = {'h':args.h, 'edges':not args.no_edges}
    elif args.kernel=="geometric":
        args_kernel = {'lbd':args.lbd}
    elif args.kernel=="nth":
        args_kernel = {'n':args.n}
    else:
        raise ValueError(f"Kernel type '{args.kernel}' not supported. Must be either 'wl', 'geometric' or 'nth'.")
    
    #processes model arguments
    if args.model=="svm":
        args_model = {'tol':args.tol}
    elif args.model=="klr":
        args_model = {'tol':args.tol, 'n_iter_max':args.n_iter_max}
    elif args.model=="krr":
        args_model = dict()
    else:
        raise ValueError(f"Model type '{args.model}' not supported. Must be either 'svm', 'krr' or 'krl'.")
    
    if args.C is not None:
        args_model['C'] = args.C
               
    return args_kernel, args_model

def load_kernel(kernel_name, split):
    if not kernel_name in ['wl', 'geometric', 'nth']:
        raise ValueError(f"Kernel type '{kernel_name}' not supported. Must be either 'wl', 'geometric' or 'nth'.")
    
    if split not in ['train', 'test']:
        raise ValueError(f"Split '{split}' not recognized. Must be either 'train' or 'test'.")
    
    return np.load(os.path.join('saved_kernels',split, f'kernel_{kernel_name}.npy')) 
        