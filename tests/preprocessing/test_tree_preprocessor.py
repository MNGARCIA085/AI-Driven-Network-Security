import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

from src.preprocessors.tree_preprocessor import TreePreprocessor


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def dummy_configs():
    global_cfg = SimpleNamespace(batch_size=32, random_state=42)
    pre_cfg = SimpleNamespace(
        path="dummy_path.csv",
        features=["f1", "f2", "f3", "Label"],
        balance_factor=0.5,
        val_size=0.2,
        scaler_type="standard",
    )
    return global_cfg, pre_cfg


@pytest.fixture
def dummy_dataframe():
    np.random.seed(0)
    df = pd.DataFrame({
        "f1": np.random.rand(100),
        "f2": np.random.rand(100),
        "f3": np.random.rand(100),
        "Label": np.random.choice(["Normal", "Bot", "Attack"], 100)
    })
    return df


# -----------------------------
# TreePreprocessor tests
# -----------------------------
@patch("pandas.read_csv")
def test_tree_preprocessor_pipeline(mock_read_csv, dummy_configs, dummy_dataframe):
    mock_read_csv.return_value = dummy_dataframe.copy()

    global_cfg, pre_cfg = dummy_configs
    pre = TreePreprocessor(global_cfg, pre_cfg)
    X_train, X_val, y_train, y_val, artifacts = pre.preprocess()

    assert isinstance(X_train, np.ndarray)
    assert len(y_train) > 0
    assert artifacts["num_classes"] >= 2


def test_tree_preprocessor_testset(dummy_configs, dummy_dataframe):
    global_cfg, pre_cfg = dummy_configs   # <-- unpack fixture here
    pre = TreePreprocessor(global_cfg, pre_cfg)
    pre.df = dummy_dataframe.copy()
    pre.combine_rare_labels().encode_labels()

    le = pre.label_encoder
    X, y = pre.preprocess_test(dummy_dataframe.copy(), le)


    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]


def test_tree_preprocessor_inference(dummy_configs, dummy_dataframe):
    global_cfg, pre_cfg = dummy_configs
    pre = TreePreprocessor(global_cfg, pre_cfg)
    X = pre.preprocess_inference(dummy_dataframe.copy())
    assert isinstance(X, np.ndarray)
    assert X.shape[1] == 3



def test_tree_preprocessor_single(dummy_configs, dummy_dataframe):
    global_cfg, pre_cfg = dummy_configs
    pre = TreePreprocessor(global_cfg, pre_cfg)
    sample = dummy_dataframe.drop(columns=["Label"]).iloc[0]
    X = pre.preprocess_single(sample)
    assert X.shape[1] == 3
