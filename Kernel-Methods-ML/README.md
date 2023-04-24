# Kaggle Challenge repository for Kernel methods for Machine Learning MVA course, spring 2023
This repository contains the graph kernels and kernel learning algorithms implemented for the Kaggle project for the course Kernel Methods for Machine Learning, spring 2023 for the Master MVA. More details about [kernels](https://github.com/JulienGaubil/MVA/tree/main/Kernel-Methods-ML/kernels) and [models](https://github.com/JulienGaubil/MVA/tree/main/Kernel-Methods-ML/models) implementations can be found in their related documentation.

# Install
```
conda create --name kernel_ml python=3.9
conda activate kernel_ml
python -m pip install -r requirements.txt
```

# Getting Started

## Training & prediction

First organize your repository as follows by creating a *data* folder and placing the appropriate training files inside.

```
.
├── kernels
│   └── ...
├── models
│   └── ...
├── data
│   └── training_data.pkl
│   └── test_data.pkl
│   └── training_labels.pkl
├── main.py
├── benchmark.py
└── ...
```

To reproduce our best result:
```
python start.py
```

To train one of the three available learning models (C-SVM, KRR, KLR) along with one of the three available kernels (WL, Geometric, N-th order random walk) and predict on the test set:
```
python main.py --kernel [kernel_name] --model [model_name] --lbd 0.1 --C 0.01
```

Valid kernel name are *wl*, *geometric* or *nth*, valid model name should be either *svm*, *krr* or *klr*. See [models](https://github.com/JulienGaubil/MVA/tree/main/Kernel-Methods-ML/models) / [kernels](https://github.com/JulienGaubil/MVA/tree/main/Kernel-Methods-ML/kernels) documentation and options help in *main.py* to add other options.

Computing kernels Geometric and N-th Order Random Walk can take up to a few hours. To use pre-computed train and test kernels, set *--load_kernel* and place the pre-computed kernels as follows:

```
.
├── saved_kernels
│   └── train
│   │   └── kernel_[kernel_name].npy
│   └── test
│   │   └── kernel_[kernel_name].npy

```

You can download our computed kernels for the default parameters using [this link](https://drive.google.com/drive/folders/12lEQkZ9FyQ-_BTZsWFYOJKtNidqCpdgE?usp=share_link).



## Benchmark

To benchmark a couple kernel+classifier using k-fold validation for the default parameters:

```
python benchmark.py --kernel [kernel_name] --model [model_name]
```

Setting *--experiment_all* benchmarks all the kernels with all the classifiers. See options in *benchmark.py* to add other options. Below are the results obtained for the precision-recall Area Under the Curve:


| Kernel / Model | SVM | KRR | KLR|
| ---------------| ----|-----|----|
| **WL**             |  61.85   | 58.81  | 60.84   |
| **N-th Order**     |  24.32  |  23.98   | 29.62 | 
| **Geometric**     | / | 20.67 | 12.66 |  





For the ROC Area Under the Curve:


| Kernel / Model | SVM | KRR | KLR|
| ---------------| ----|-----|----|
| **WL**             |  89.88   | 85.26 |  87.69    |
| **N-th Order**     |  72.66  | 64.55 |  70.99  | 
| **Geometric**     | / |  70.83 |  52.87 |  



# References


We implemented the Geometric graph kernel following:
```BibTex
@article{vishwanathan2010graph,
title = {Graph Kernels},
author = {Vishwanathan, S. V. N. and Schraudolph, Nicol N. and Kondor, Risi and Borgwardt, Karsten M.},
journal = {Journal of Machine Learning Research},
year = {2010}
}
```


We implemented the Weisfeiler-Lehman graph kernel following the following article:
```BibTex
@article{shervashidze2011weisfeiler,
  title = {Weisfeiler-Lehman graph kernels},
  author = {Shervashidze, Nino and Schweitzer, Pascal and Van Leeuwen, Erik Jan and Mehlhorn, Kurt and Borgwardt, Karsten M},
  journal = {Journal of Machine Learning Research},
  year = {2011}
}
```
