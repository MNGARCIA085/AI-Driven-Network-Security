import pytest
import numpy as np
from src.tuning.nn import NNTuner
from src.tuning.tree import TreeTuner
from src.tuning.factory import TunerFactory


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



@pytest.fixture
def preprocessor():
    class DummyPreprocessor:
        num_classes = 3
    return DummyPreprocessor()


# ---------- TESTS ----------
def test_get_tree_tuner(cfg, sample_data, preprocessor):
    X_train, y_train, X_val, y_val = sample_data
    tuner = TunerFactory.get_tuner("tree", cfg, X_train, y_train, X_val, y_val, preprocessor)
    assert isinstance(tuner, TreeTuner)


def test_get_nn_tuner(cfg, sample_data, preprocessor):
    X_train, y_train, X_val, y_val = sample_data
    tuner = TunerFactory.get_tuner("nn", cfg, X_train, y_train, X_val, y_val, preprocessor)
    assert isinstance(tuner, NNTuner)
    assert tuner.num_classes == preprocessor.num_classes


def test_get_tuner_invalid(cfg, sample_data, preprocessor):
    X_train, y_train, X_val, y_val = sample_data
    with pytest.raises(ValueError):
        TunerFactory.get_tuner("invalid_type", cfg, X_train, y_train, X_val, y_val, preprocessor)


#pytest -v -s tests/tuning/test_factory.py