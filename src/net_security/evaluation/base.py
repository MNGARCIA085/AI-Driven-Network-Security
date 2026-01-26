import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score



class Evaluator:
    def __init__(self, average='weighted'): # y_probs;
        self.average = average

    def compute_metrics(self, y_true, y_pred, total_loss=None, total_samples=None):
        #Compute common metrics for both neural nets and tree-based models.
        metrics = {}

        # Loss (only if available)
        if total_loss is not None and total_samples is not None:
            metrics["avg_loss"] = total_loss / total_samples
    
                    
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["f1"] = f1_score(y_true, y_pred, average=self.average, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average=self.average, zero_division=0)
        metrics["precision"] = precision_score(y_true, y_pred, average=self.average, zero_division=0)
        
        return metrics

    # if i need probs i can pass them...........

    
    