import numpy as np
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    accuracy_score, 
    classification_report,
    multilabel_confusion_matrix
)

def compute_multilabel_metrics(eval_pred, threshold: float = 0.5):
    """
    Computes metrics for multi-label classification.
    Expects logits from the model, applies sigmoid, and thresholds.
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    # Apply sigmoid using numpy
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs >= threshold).astype(int)
    
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    macro_precision = precision_score(labels, predictions, average='macro', zero_division=0)
    macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)
    exact_match = accuracy_score(labels, predictions) # Exact Match Ratio
    
    metrics_dict = {
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'exact_match_ratio': exact_match
    }

    # Calculate per-class F1 if we want to log them without explicit names (using index)
    # They can be mapped later, or we can just provide a standalone per-class function
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    for i, f1 in enumerate(per_class_f1):
        metrics_dict[f'f1_class_{i}'] = f1
        
    return metrics_dict

def get_per_class_report(labels, predictions, label_names):
    """
    Returns a dictionary of per-class F1, Precision, and Recall scores.
    """
    f1s = f1_score(labels, predictions, average=None, zero_division=0)
    precs = precision_score(labels, predictions, average=None, zero_division=0)
    recs = recall_score(labels, predictions, average=None, zero_division=0)
    
    report = {}
    for i, name in enumerate(label_names):
        report[name] = {
            'f1': f1s[i],
            'precision': precs[i],
            'recall': recs[i]
        }
    return report


def find_optimal_thresholds(logits, labels, label_names, search_range=None):
    """
    Search per-class optimal thresholds that maximize Macro-F1.
    Returns (list of thresholds, best macro_f1).
    """
    if search_range is None:
        search_range = np.arange(0.1, 0.9, 0.05)
    
    probs = 1 / (1 + np.exp(-logits))
    n_classes = labels.shape[1]
    best_thresholds = [0.5] * n_classes
    
    for i in range(n_classes):
        best_f1 = 0.0
        for thresh in search_range:
            preds_i = (probs[:, i] >= thresh).astype(int)
            f1_i = f1_score(labels[:, i], preds_i, zero_division=0)
            if f1_i > best_f1:
                best_f1 = f1_i
                best_thresholds[i] = round(float(thresh), 2)
    
    # Calculate overall Macro-F1 with optimal thresholds
    final_preds = np.zeros_like(labels)
    for i in range(n_classes):
        final_preds[:, i] = (probs[:, i] >= best_thresholds[i]).astype(int)
    
    optimized_f1 = f1_score(labels, final_preds, average='macro', zero_division=0)
    
    print(f"  Per-class optimal thresholds:")
    for name, thresh in zip(label_names, best_thresholds):
        print(f"    {name}: {thresh}")
    
    # Classification Report
    class_report = classification_report(labels, final_preds, target_names=label_names, zero_division=0, digits=4)
    
    # Confusion Matrix
    conf_matrix = multilabel_confusion_matrix(labels, final_preds)
    
    return best_thresholds, optimized_f1, class_report, conf_matrix
