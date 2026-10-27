"""
Modular Evaluation Metrics and Calibration Analysis Module.

Provides reproducible calculation of:
- Balanced Accuracy
- Macro F1-Score
- Multiclass One-vs-Rest Macro AUROC
- 7x7 Confusion Matrix
- Per-Class Sensitivity (Recall) and Specificity
- Expected Calibration Error (ECE) and Reliability Diagram Binning Data
"""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def compute_ece(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) and binning statistics.
    
    Args:
        y_true (np.ndarray): True integer class labels [N].
        y_prob (np.ndarray): Predicted class probabilities [N, K].
        n_bins (int): Number of equal-width probability confidence bins (default 10).
        
    Returns:
        tuple: (ece_value, bin_accuracies, bin_confidences, bin_counts)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    confs = np.max(y_prob, axis=1)
    preds = np.argmax(y_prob, axis=1)
    accuracies = (preds == y_true).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Include lower bound for first bin
        if i == 0:
            in_bin = (confs >= bin_lower) & (confs <= bin_upper)
        else:
            in_bin = (confs > bin_lower) & (confs <= bin_upper)
            
        bin_size = np.sum(in_bin)
        bin_counts.append(int(bin_size))
        
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confs[in_bin])
            
            bin_accuracies.append(float(bin_acc))
            bin_confidences.append(float(bin_conf))
            
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)
            
    return float(ece), bin_accuracies, bin_confidences, bin_counts


def compute_per_class_sensitivity_specificity(cm):
    """
    Computes per-class sensitivity (recall) and specificity from a KxK confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix of shape [K, K] where row=true, col=pred.
        
    Returns:
        tuple: (sensitivities_list, specificities_list)
    """
    K = cm.shape[0]
    sensitivities = []
    specificities = []
    
    total_samples = np.sum(cm)
    
    for c in range(K):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = total_samples - (tp + fn + fp)
        
        # Sensitivity = TP / (TP + FN)
        if (tp + fn) > 0:
            sens = tp / (tp + fn)
        else:
            sens = np.nan
            
        # Specificity = TN / (TN + FP)
        if (tn + fp) > 0:
            spec = tn / (tn + fp)
        else:
            spec = np.nan
            
        sensitivities.append(float(sens) if not np.isnan(sens) else np.nan)
        specificities.append(float(spec) if not np.isnan(spec) else np.nan)
        
    return sensitivities, specificities


def calculate_all_metrics(y_true, y_prob, num_classes=7, n_bins=10):
    """
    Calculates comprehensive evaluation metrics for multiclass classification.
    
    Args:
        y_true (np.ndarray): Ground truth integer labels [N].
        y_prob (np.ndarray): Model output softmax probabilities [N, K].
        num_classes (int): Total number of target classes (default 7).
        n_bins (int): Bins for ECE calculation (default 10).
        
    Returns:
        dict: Complete metrics dictionary.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    
    # 1. Balanced Accuracy
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    # 2. Macro F1-Score
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=np.nan))
    
    # 3. Macro AUROC (Multiclass One-vs-Rest, computed per-class for robust numerical stability)
    try:
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        if num_classes == 2 and y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        class_aurocs = []
        for c in range(num_classes):
            if c < y_prob.shape[1] and len(np.unique(y_true_bin[:, c])) > 1:
                auc_c = roc_auc_score(y_true_bin[:, c], y_prob[:, c])
                class_aurocs.append(auc_c)
        
        macro_auroc = float(np.mean(class_aurocs)) if class_aurocs else np.nan
    except Exception:
        macro_auroc = np.nan
            
    # 4. Confusion Matrix [K, K]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # 5. Per-class Sensitivity & Specificity
    sensitivities, specificities = compute_per_class_sensitivity_specificity(cm)
    
    # 6. Expected Calibration Error (ECE)
    ece, bin_accs, bin_confs, bin_counts = compute_ece(y_true, y_prob, n_bins=n_bins)
    
    return {
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'macro_auroc': macro_auroc,
        'confusion_matrix': cm,
        'sensitivities': sensitivities,
        'specificities': specificities,
        'ece': ece,
        'bin_accuracies': bin_accs,
        'bin_confidences': bin_confs,
        'bin_counts': bin_counts,
        'n_bins': n_bins,
        'num_classes': num_classes,
        'present_classes': list(present_classes)
    }
