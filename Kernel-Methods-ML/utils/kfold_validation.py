import numpy as np
from utils.metrics import *
from utils.tools import *
from tqdm import tqdm
from models.svm import CSVM
from sklearn.metrics import roc_auc_score



def create_folds(X, labels, k, shuffle=False):
    """
    Creates folds from given data.
    Parameters
    ----------
    X: list(object).
        Training data.
    labels: np.array (shape=(n,)).
        Binary ground-truth labels {0,1} for the samples.
    k: int.
        Number of folds to use.
    shuffle: (bool).
        Whether to shuffle the data or not.
    Returns
    ----------
    X_folds: list(list)
        List of k folds of X.
    labels_folds: list(np.array)
        List of k folds of labels.
    idx_folds: list(np.array)
        List of k folds of indexes in original X of the elements in the fold.
    """
    assert len(labels)==len(X)
    n = len(X)
    n_ = n//k
    X_folds, labels_folds, idx_folds = list(), list(), list()

    idx_perm = np.random.permutation(np.arange(n)) if shuffle else np.arange(len(X))

    X = X[idx_perm]
    labels = labels[idx_perm]   
    
    for i in range(k):
        if i < k-1:
            X_folds.append(X[i*n_:(i+1)*n_])
            labels_folds.append(labels[i*n_:(i+1)*n_])
            idx_folds.append(idx_perm[i*n_:(i+1)*n_])
        else:
            X_folds.append(X[i*n_:])  
            labels_folds.append(labels[i*n_:])
            idx_folds.append(idx_perm[i*n_:]) 

    return X_folds, labels_folds, idx_folds



def merge_folds(X_folds, labels_folds, idx_folds, i):
    """
    Merges all folds in X_folds and labels_folds except for the one with index i.
    Parameters
    ----------
    X_folds: list(list)
        List of k folds of X.
    labels_folds: list(np.array)
        List of k folds of labels.
    idx_folds: list(np.array)
        List of k folds of indexes in original X of the elements in the fold.
    i: int.
        index of the fold not to merge.
    Returns
    ----------
    merged_X: list(object).
        Training data.
    merged_labels_folds: np.array (shape=(n,)).
        Binary ground-truth labels {0,1} for the samples.
    merged_idx_folds: np.array (shape=(n,)).
        Indexes in the original dataset of the merged samples.
    """
    merged_X = np.array([], dtype=object)
    merged_labels_folds = np.array([])
    merged_idx_folds = np.array([], dtype=np.int32)
    for k in range(len(X_folds)):
        if k!=i:
            merged_X = np.concatenate((merged_X, X_folds[k]))
            merged_labels_folds = np.concatenate([merged_labels_folds, labels_folds[k]], axis=0)
            merged_idx_folds = np.concatenate([merged_idx_folds, idx_folds[k]], axis=0)

    return np.array(merged_X, dtype=object), merged_labels_folds, merged_idx_folds

    
def kfold_validation(estimator, labels, kfolds, X=None, K=None):
    """
    Creates folds from given data.
    Parameters
    ----------
    labels: np.array (shape=(n,)).
        Binary ground-truth labels {0,1} for the samples.
    kfolds: int.
        Number of folds to use.
    shuffle: (bool).
        Whether to shuffle the data or not.
    X: list(object).
        Optional. Training data.
    args: dict.
        Optional. Contains arguments for experiments such as dirpath, filename to save the Gram matrix of the kernel.
    Returns
    ----------
    auc, precision, recall, accuracy, f1: lists(float)
        List of statistics collected on each val fold during K-fold validation.   
    """
    X = X if X is not None else estimator.X_tr
    K = K if K is not None else estimator.kernel.compute_K(X)

    X_folds, labels_folds, idx_folds = create_folds(X,labels,kfolds,shuffle=True)

    auc, accuracy, recall, precision, auc_roc =  np.zeros(kfolds), np.zeros(kfolds), np.zeros(kfolds), np.zeros(kfolds), np.zeros(kfolds)
    
    for k in range(kfolds):

        X_val, labels_val, idx_val = X_folds[k], labels_folds[k], idx_folds[k]
        X_tr, labels_tr, idx_tr = merge_folds(X_folds, labels_folds, idx_folds, k)

        #updates kernel data and Gram matrix with folds
        estimator.kernel.X_tr = X_tr
        estimator.kernel.X_te = X_val
        estimator.kernel.K_tr = K[idx_tr,:][:,idx_tr] #extracts pre-computed train kernel
        estimator.kernel.K_te = K[idx_folds[k],:][:,idx_tr] #extracts pre-computed test kernel (n_test,n_train)

        estimator.train(labels_tr)

        #evaluates
        idx = idx_tr[estimator.active_idx] if isinstance(estimator, CSVM) else idx_tr
        preds_tr = estimator.predict_values(Kx=K[idx_tr,:][:,idx])
        preds_val = estimator.predict_values(test=True)
        pred_classes_val = np.sign(preds_val)

        auc_val = compute_auc_precision_recall(labels_val, preds_val)
        auc_train = compute_auc_precision_recall(labels_tr, preds_tr)
        auc_roc_val = roc_auc_score(labels_val, preds_val)
        auc_roc_tr = roc_auc_score(labels_tr, preds_tr)
        prec, rec, acc = compute_precision_recall(labels_val, pred_classes_val)
        print(f'[{k+1}/{kfolds}] Accuracy : {round(acc, 4)}, Precision: {round(prec,4)}, Recall : {round(rec,4)}, AUC : {round(auc_val,4)}, AUC ROC : {round(auc_roc_val,4)}')
        print(f'[{k+1}/{kfolds}] AUC train : {round(auc_train,4)}, AUC train ROC: {round(auc_roc_tr,4)}')
        print('')
        precision[k] = prec
        recall[k] = rec
        accuracy[k] = acc
        auc[k] = auc_val
        auc_roc[k] = auc_roc_val

    f1 = 2 * precision * recall / (precision + recall)
    print('avg auc: {}, avg auc roc: {}, avg precision: {}, avg recall: {}, avg accuracy: {}, avg f1: {}.'.format(
       round(auc.mean(),4), round(auc_roc.mean(),4), round(precision.mean(),4), round(recall.mean(),4), round(accuracy.mean(),4), round(f1.mean(),4)))
    
    print('std auc: {}, std auc roc: {}, std precision: {}, std recall: {}, std accuracy: {}, std f1: {}.'.format(
        round(auc.std(),4), round(auc_roc.std(),4), round(precision.std(),4), round(recall.std(),4), round(accuracy.std(),4), round(f1.std(),4)))
    print('')
    return auc, precision, recall, accuracy, f1
