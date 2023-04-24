import numpy as np
import networkx as nx
from kernels.base_kernel import AbstractKernel
from copy import deepcopy
import time

class WL(AbstractKernel):

    def __init__(self, K_tr=None, K_te=None, X_tr=None, X_te=None, h = 3, edges=True, verbose=True):
        self.h = h
        self.unique_labels_h = []
        self.unique_labels_count = 0
        self.edges = edges
        super().__init__(K_tr, K_te, X_tr, X_te, verbose)

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
    
    
    def prepare_kernels(self, X):
        """
        Weisfeiler-Lehman subtree kernel
        """
        graphs = deepcopy(X)
        self.unique_labels_h = []
        # Initialization
        unique_labels_0 = []

        for graph in graphs:
            unique_labels_0 += list(nx.get_node_attributes(
                graph, name='labels').values())
        unique_labels_0 = np.unique(unique_labels_0)
        unique_labels = {label: i for i, label in enumerate(
            unique_labels_0)} 

        feat_vector_0 = np.zeros((len(graphs), len(unique_labels_0)), dtype=int)
        for i, graph in enumerate(graphs):
            for _, label in graph.nodes(data='labels'):
                feat_vector_0[i, int(unique_labels[label])] += 1
        feat_vector = feat_vector_0 # feature vector for initial depth
        self.unique_labels_h.append(unique_labels)  # Store initial labels

        for _ in range(self.h):
            unique_labels = dict()
            count = 0

            for id, graph in enumerate(graphs):
                gc = graph.copy()
                for node, label in graph.nodes(data='labels'):
                    
                    # multiset labelling
                    added_label = [str(graph.nodes[neighbor]['labels']) for neighbor in graph.neighbors(node)]
                    if self.edges:  # Consider edges in the labelling
                        added_label += [str(graph.edges[node, neighbor]['labels'][0]) for neighbor in graph.neighbors(node)]

                    #sorting each multiset
                    added_label = sorted(added_label)

                    new_label = label + ''.join(str(x) for x in added_label) 

                    # Label compression
                    if new_label in unique_labels.keys(): #test if already seen in depth h
                        compressed_label = unique_labels[new_label]
                    else:
                        unique_labels.update({new_label: count})
                        compressed_label = count
                        count += 1
                    gc.nodes[node]['labels'] = str(compressed_label) # relabel the copy
                # change for future iteration    
                graphs[id] = gc
            
            # keep track of labels for each depth
            self.unique_labels_h.append(unique_labels)

            # Compute feature vector at step h
            feat_vector_h = np.zeros((len(graphs), count), dtype=int)
            for i, graph in enumerate(graphs):
                for _, label in graph.nodes(data='labels'):
                    feat_vector_h[i, int(label)] += 1
            
            feat_vector = np.hstack((feat_vector, feat_vector_h)) # add new features to the previous ones

        return feat_vector


    def compute_K(self, X1, X2=None):
        if self.verbose:
            start = time.time()
            print('Starts computing kernel')   
        if X2 is None:
            features = self.prepare_kernels(X1)
            Kx = features.dot(features.T)
        else:
            n = len(X1)
            features = self.prepare_kernels(np.concatenate((X1,X2)))
            features1 = features[:n]
            features2 = features[n:]
            Kx = features1.dot(features2.T)
        
        if self.verbose:
            end = time.time()
            print('Kernel computation finished in {0:.2f}'.format(end-start))   

        return Kx
    
    def evaluate(self, x,y):
        k = self.compute_K([x],[y]).item()
        return k
        






