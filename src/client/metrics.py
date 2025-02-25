import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from typing import Tuple


def evaluation_metrics(y_true: np.ndarray, classes: np.ndarray, predicted_test: np.ndarray) -> Tuple[float, float, float, float]:
    accuracy = accuracy_score(y_true, classes)
    
    cnf_matrix = confusion_matrix(y_true, classes)
    
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    TPR = TP/(TP+FN)
    tpr_mean = np.nanmean(TPR)
    
    FPR = FP/(FP+TN)
    fpr_mean = np.nanmean(FPR)
    
    f1 = f1_score(y_true, classes, average='weighted')
    
    auc = roc_auc_score(y_true, predicted_test, multi_class='ovr')
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'TPR: {tpr_mean:.4f}')
    print(f'FPR: {fpr_mean:.4f}')
    print(f'F1 score: {f1:.4f}')
    print(f'AUC Score: {auc:.4f}')
    
    return tpr_mean, fpr_mean, f1, auc