from sklearn import metrics
import numpy as np

def compute_auc_precision_recall(labels, preds):
    """
        Computes AUC for precision-recall curve from regressed values.
        Parameters
        ----------
        labels: np array (shape=(n,)).
            Binary ground-truth labels {0,1} for the samples.
        preds: np array (shape=(n,)).
            Values regressed for the samples.
        Returns
        ----------
        auc: float.
            Area Under the Curve precision-recall for the predicted values.
    """
    precision, recall, thresholds = metrics.precision_recall_curve(labels, preds)
    auc = metrics.auc(recall, precision)
    return auc

def compute_precision_recall(labels,pred_classes):
    """
        Computes AUC for precision-recall curve from regressed values.
        Parameters
        ----------
        labels: np array (shape=(n,)).
            Binary ground-truth labels {0,1} for the samples.
        pred_classes: np array (shape=(n,)).
            Binary labels {0,1} predicted for the samples.
        Returns
        ----------
        precision: float.
            Precision for the predicted classes.
        recall: float.
            Recall for the predicted classes.
        accuracy: float.
            Accuracy for the predicted classes.
    """
    tp = np.sum((pred_classes == 1.) * (labels == 1.))
    fn = np.sum((pred_classes == -1.) * (labels == 1.))
    fp = np.sum((pred_classes == 1.) * (labels == -1.))
    recall = tp/(fn + tp)
    precision = tp/(fp + tp)
    accuracy = np.sum(labels == pred_classes)/len(labels)
    return precision, recall, accuracy
