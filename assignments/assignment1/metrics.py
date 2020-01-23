import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    print(classification_report(prediction, ground_truth))
    print(accuracy_score(prediction, ground_truth))
    print(confusion_matrix(prediction, ground_truth))
    
    tn_m, fp_m, fn_m, tp_m = confusion_matrix(prediction, ground_truth).ravel()
    print(f"from confusion matrix: TN: {tn_m} FP: {fp_m} FN: {fn_m} TP: {tp_m}")

    tp = np.sum(prediction & ground_truth)
    fn = np.sum(prediction & np.logical_not(ground_truth))
    fp = np.sum(np.logical_not(prediction) & ground_truth)
    tn = np.sum(np.logical_not(prediction) & np.logical_not(ground_truth))
    print(f"                       TN: {tn} FP: {fp} FN: {fn} TP: {tp}")

    precision = tp / (tp + fp)
    recall = tp / (tp  + fn)
    accuracy = (tn + tp) / (tn + fp + fn + tp) 
    
    #precision = np.sum(prediction) / prediction.shape[0]
    #recall = np.sum(prediction) / np.sum(ground_truth)
    #accuracy = np.sum(np.equal(prediction, ground_truth)) / prediction.shape[0]
    f1 = 2.0 * precision * recall / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return 0
