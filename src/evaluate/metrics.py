import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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
    
    return {
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'exact_match_ratio': exact_match
    }
