from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def evaluate_accuracy(predictions, targets):
    accuracy = np.mean(predictions == targets)
    return accuracy
    

def evaluate_stat(predictions, targets, prob_pred):
    """ compute evaluation statistics"""
    accuracy = evaluate_accuracy(predictions, targets)
    tp = np.sum(predictions[np.where(targets)])
    tn = np.sum(predictions[np.where(targets==0)]==0)
    sensitivity = tp / np.sum(targets)
    specificity = tn / np.sum(targets==0)
    precision = tp / np.sum(predictions)
    recall = tn / np.sum(predictions == 0)
    fpr, tpr, thresholds = roc_curve(targets, prob_pred, pos_label=0)
    auc_value = auc(fpr, tpr)
    return (accuracy, sensitivity, specificity, precision, recall, auc_value)

def evaluate_confusion(predictions, targets):
    """ compute confusion matrix"""
    cm = confusion_matrix(targets, predictions)
    return cm

