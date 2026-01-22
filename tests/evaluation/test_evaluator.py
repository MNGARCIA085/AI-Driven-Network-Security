from src.evaluation.base import Evaluator
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# ------------------------
# Basic Evaluator test
# ------------------------
def test_evaluator_metrics(y_true, y_pred):
    evaluator = Evaluator(average='weighted')
    metrics = evaluator.compute_metrics(y_true, y_pred)

    # Check keys
    for key in ["accuracy", "f1", "recall", "precision"]:
        assert key in metrics

    # Check values against sklearn directly
    assert metrics["accuracy"] == accuracy_score(y_true, y_pred)
    assert metrics["f1"] == f1_score(y_true, y_pred, average='weighted', zero_division=0)
    assert metrics["recall"] == recall_score(y_true, y_pred, average='weighted', zero_division=0)
    assert metrics["precision"] == precision_score(y_true, y_pred, average='weighted', zero_division=0)

# ------------------------
# Evaluator with total_loss
# ------------------------
def test_evaluator_with_loss(y_true, y_pred):
    evaluator = Evaluator(average='weighted')
    total_loss = 2.5
    total_samples = len(y_true)
    metrics = evaluator.compute_metrics(y_true, y_pred, total_loss=total_loss, total_samples=total_samples)

    assert "avg_loss" in metrics
    assert metrics["avg_loss"] == total_loss / total_samples


# ------------------------
# Evaluator single-sample edge case
# ------------------------
def test_evaluator_single_sample():
    y_true = np.array([1])
    y_pred = np.array([1])
    evaluator = Evaluator(average='weighted')
    metrics = evaluator.compute_metrics(y_true, y_pred)
    
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0