import pytest
import numpy as np
import ray

from src.tuning.base import BaseTuner
from src.tuning.tree import TreeTuner

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
    num_classes = 10 # maybe later len(np.unique(y_train))
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


# ---------- BaseTuner ----------
def test_base_tuner_puts_data(cfg, sample_data):
    X_train, y_train, X_val, y_val, num_classes = sample_data
    tuner = TreeTuner(cfg, X_train, y_train, X_val, y_val, num_classes)
    assert isinstance(tuner.X_train_id, ray.ObjectRef)
    assert tuner.average == "weighted"
