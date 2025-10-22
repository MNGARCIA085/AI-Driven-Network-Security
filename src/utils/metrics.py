from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def compute_metrics(labels, preds, total_loss=None, total_samples=None, average="weighted"):
    """
    Compute common metrics for both neural nets and tree-based models.
    - total_loss and total_samples are optional (only for models with explicit loss)
    """
    metrics = {}
    
    # Loss (only if available)
    if total_loss is not None and total_samples is not None:
        metrics["avg_loss"] = total_loss / total_samples
    
    metrics["accuracy"] = accuracy_score(labels, preds)
    metrics["f1"] = f1_score(labels, preds, average=average, zero_division=0)
    metrics["recall"] = recall_score(labels, preds, average=average, zero_division=0)
    metrics["precision"] = precision_score(labels, preds, average=average, zero_division=0)
    
    return metrics



