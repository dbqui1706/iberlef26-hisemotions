import numpy as np
from src.evaluate.metrics import compute_multilabel_metrics

def test_compute_metrics():
    # Simulate an EvalPrediction object containing predictions (logits) and true labels
    class MockEvalPrediction:
        def __init__(self, predictions, label_ids):
            self.predictions = predictions
            self.label_ids = label_ids
            
    # true: [joy, hope], pred: [joy, hope] (high logit > 0)
    true_labels = np.array([[0, 0, 1, 0, 0, 1]])
    logits = np.array([[-1.0, -2.0, 1.5, -0.5, -3.0, 0.8]]) 
    
    eval_pred = MockEvalPrediction(logits, true_labels)
    metrics = compute_multilabel_metrics(eval_pred)
    
    assert 'macro_f1' in metrics
    assert metrics['macro_f1'] == 1.0 # Perfect match
