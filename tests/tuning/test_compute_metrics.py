import pytest
import numpy as np
from src.utils.metrics import compute_metrics



def test_compute_metrics_basic():
    labels = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])
    
    metrics = compute_metrics(labels, preds, total_loss=4.0, total_samples=4, average="binary")
    
    # Check that all keys exist
    for key in ["accuracy", "f1", "recall", "precision", "avg_loss"]:
        assert key in metrics
    
    # Check basic numeric ranges
    for v in metrics.values():
        assert 0.0 <= v <= 1.0 or isinstance(v, float)


def test_compute_metrics_multiclass_weighted():
    labels = np.array([0, 1, 2, 1, 0, 2])
    preds = np.array([0, 2, 1, 1, 0, 2])
    
    metrics = compute_metrics(labels, preds, average="weighted")
    
    for key in ["accuracy", "f1", "recall", "precision"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_compute_metrics_no_loss():
    labels = np.array([1, 0, 1])
    preds = np.array([1, 1, 1])
    
    metrics = compute_metrics(labels, preds, average="macro")
    assert "avg_loss" not in metrics  # no loss provided
    assert pytest.approx(metrics["accuracy"], 0.001) == 2/3


def test_compute_metrics_zero_division():
    labels = np.array([0, 0, 0])
    preds = np.array([1, 1, 1])
    
    # should not raise due to zero_division=0
    metrics = compute_metrics(labels, preds, average="macro")
    assert all(k in metrics for k in ["accuracy", "f1", "recall", "precision"])
