import pytest
import numpy as np
import torch
import ray

from src.tuning.tree import TreeTuner
from src.models.tree import TreeModel
from src.utils.results import Metrics



@pytest.fixture(scope="module", autouse=True)
def setup_ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def sample_data():
    X_train = np.random.rand(20, 4)
    y_train = np.random.randint(0, 2, 20)
    X_val = np.random.rand(10, 4)
    y_val = np.random.randint(0, 2, 10)
    num_classes = 10
    return X_train, y_train, X_val, y_val, num_classes


@pytest.fixture
def cfg():
    class DummyCfg:
        average = "weighted"
        epochs = 2
        epochs_trials = 1
        hidden1 = [8, 16]
        hidden2 = [4, 8]
        batch_size = [4]
        lr = type("lr", (), {"min": 1e-4, "max": 1e-2})
        criterion = ["gini", "entropy"]
        max_depth = type("range", (), {"min": 2, "max": 4})
        min_samples_split = type("range", (), {"min": 2, "max": 5})
    return DummyCfg()



# ------------ Tests--------------
def test_tree_tuner_get_config(cfg, sample_data):
    X_train, y_train, X_val, y_val, num_classes = sample_data
    tuner = TreeTuner(cfg, X_train, y_train, X_val, y_val, num_classes)
    config = tuner.get_tune_config()
    assert "model.criterion" in config
    assert "model.max_depth" in config
    #print(config, flush=True)




"""
belongs to train!!!!!!!!!!

def test_tree_tuner_train_best_model(cfg, sample_data, monkeypatch):
    X_train, y_train, X_val, y_val, num_classes = sample_data
    tuner = TreeTuner(cfg, X_train, y_train, X_val, y_val, num_classes)

    class DummyModel:
        def fit(self, X, y): pass
        def predict(self, X): return np.zeros(len(X))
        def predict_proba(self, X): return np.zeros((len(X), 2))

    monkeypatch.setattr("src.models.tree.TreeModel", DummyModel)
    results = tuner.train_best_model({
        "criterion": "gini",
        "max_depth": 3,
        "min_samples_split": 2
    })
    assert hasattr(results, "val")
    assert isinstance(results.val.metrics, Metrics)
"""