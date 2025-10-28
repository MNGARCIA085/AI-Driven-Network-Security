import pytest
import numpy as np
import torch
import ray

from src.tuning.base import BaseTuner
from src.tuning.nn import NNTuner
from src.models.nnet import NNModel


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
    return X_train, y_train, X_val, y_val


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



# ---------- TESTS ----------

def test_nn_tuner_init(cfg, sample_data):
    X_train, y_train, X_val, y_val = sample_data
    tuner = NNTuner(cfg, X_train, y_train, X_val, y_val, num_classes=2)
    assert tuner.input_size == X_train.shape[1]
    assert tuner.num_classes == 2
    assert tuner.device in ["cpu", "cuda"]


def test_nn_tuner_get_config(cfg, sample_data):
    X_train, y_train, X_val, y_val = sample_data
    tuner = NNTuner(cfg, X_train, y_train, X_val, y_val, num_classes=2)
    config = tuner.get_tune_config()
    assert "lr" in config and "hidden1" in config


def test_nn_tuner_train_one_epoch(cfg, sample_data):
    X_train, y_train, X_val, y_val = sample_data
    tuner = NNTuner(cfg, X_train, y_train, X_val, y_val, num_classes=2)
    model = torch.nn.Linear(4, 2)
    loader, _ = tuner.create_loaders(batch_size=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss, acc = tuner.train_one_epoch(model, loader, opt, loss_fn)
    assert loss > 0
    assert 0 <= acc <= 1

