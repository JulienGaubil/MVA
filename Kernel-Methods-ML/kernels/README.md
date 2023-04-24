# Kernels documentation

The graph kernels implemented are:

- N-th Order Random Walk Kernel
- Geometric Random Walk Kernel
- Weisfeiler-Lehman Kernel

All the kernels implemented are subclasses of the abstract class `AbstractKernel` that provides a framework with method `compute_K` that enables to compute the kernel given one or two sets of graphs as an input. Kernels can either be computed entry by entry, or in a single pass.

If a single set of graphs $\{x_1, ..., x_n \}$ is provided as an input, `compute_K` returns $K=(k(x_i, x_j))$ and saves half of the computation by leveraging the symmetrical structure of $ K $. 

If two sets of graphs ${x_1, ..., x_n \}$ and $\{y_1, ..., y_n \}$ are provided as an input, `compute_K` returns $K=(k(x_i, y_j))$ by computing every entry.


## N-th Order Kernel

The N-th order random walk kernel is computed by $k(G_1, G_2) = \mathbf{1}^T A^N \mathbf{1}$ where $A$ is the adjacency matrix of the direct product graph of labelled graphs $G_1$ and $G_2$.

### Adjacency matrix computation

The method `prepare_kernels` takes as an input a list of graphs and returns for each graph $G$ a its adjacency matrices $A_G^{(l_1,l_2)}$ filtered by the pair of node labels $(l_1,l_2)$: 

$$\left(A_G^{(l_1,l_2)}\right)_{i,j}= 1_{(label(i)=l_1, label(j)=l_2)} \left(A_G\right)_{i,j}$$


This is useful to compute the adjacency matrix $A$ of the direct product graph of two graphs $G_1, G_2$: let $L$ the set of common node labels between $G_1$ and $G_2$, then:

$$A=\sum_{l_1,l_2\in L} A_{G_1}^{(l_1,l_2)}\otimes A_{G_2}^{(l_1,l_2)}$$

where $\otimes$ denotes the Kronecker product.

### Kernel evaluation

We leverage potentially available GPUs using PyTorch to speed up the computation. When no GPUs are available, we further accelerate the computation by parallelizing the computation accross all available CPUs using joblib with the argument `multi` in `compute_K`.

## Geometric Kernel
Following the Graph Kernel paper below, we use the Conjugate Gradient method by first solving the linear system: $$(I_d - \lambda A)x=\mathbf{1}$$, and then computing $k(G_1, G_2) = \mathbf{1}^T x$ where $A$ is the adjacency matrix of the direct product graph of labelled graphs $G_1$ and $G_2$.

To invert the linear system, we define a linear operator leveraging the mixed Kronecker matrix-vector product. The method `prepare_kernels` is the same as for the N-th Order Kernel and enables to define this linear operator.


```BibTex
@article{vishwanathan2010graph,
title = {Graph Kernels},
author = {Vishwanathan, S. V. N. and Schraudolph, Nicol N. and Kondor, Risi and Borgwardt, Karsten M.},
journal = {Journal of Machine Learning Research},
year = {2010}
}
```



## Weisfeiler-Lehman subtree kernel

Following the Weisfeiler-Lehman Graph kernel paper below, we create a multiset for each node and its neighborhood, and we apply an hash function to this multiset. Then we compress the label and we can iterate this procedure several time. Each time we count the number of the same multiset appearing in the graph, and we put this data in a feature vector. Concatening these vectors for each depth, we obtain a global feature vector and the kernel is calculated via the standard inner product between the feature vectors of the two graphs.

Additionally, we can leverage the information in the edges by adding in the multiset the edge labels of the neighborhood. This is an option given by the boolean `edges`.

The method `prepare_kernels` takes as an input a list of graphs and returns for each graph $G$ the Weisfeiler-Lehman feature vector. 

```Bibtex
@article{journals/jmlr/ShervashidzeSLMB11,  
  title = {Weisfeiler-Lehman Graph Kernels.},
  author = {Shervashidze, Nino and Schweitzer, Pascal and van Leeuwen, Erik Jan and Mehlhorn, Kurt and Borgwardt, Karsten M.},
  journal = {J. Mach. Learn. Res.},
  year = 2011
}

```