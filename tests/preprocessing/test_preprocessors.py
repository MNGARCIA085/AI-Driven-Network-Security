import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from types import SimpleNamespace
from unittest.mock import patch

from src.preprocessors.base import BasePreprocessor
from src.preprocessors.nn_preprocessor import NNPreprocessor
from src.preprocessors.tree_preprocessor import TreePreprocessor
from src.preprocessors.factory import PreprocessorFactory

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
# BasePreprocessor structure
# -----------------------------
def test_base_preprocessor_abstract_methods(dummy_configs):
    global_cfg, pre_cfg = dummy_configs
    with pytest.raises(TypeError):
        _ = BasePreprocessor(global_cfg, pre_cfg)


# -----------------------------
# NNPreprocessor tests
# -----------------------------
@patch("pandas.read_csv")
@pytest.mark.nn
def test_nn_preprocessor_pipeline(mock_read_csv, dummy_configs, dummy_dataframe):
    mock_read_csv.return_value = dummy_dataframe.copy()

    global_cfg, pre_cfg = dummy_configs
    pre = NNPreprocessor(global_cfg, pre_cfg)
    X_train, X_val, y_train, y_val, artifacts = pre.preprocess()

    assert isinstance(X_train, np.ndarray)
    assert X_train.shape[1] == 3
    assert len(y_train) > 0
    assert artifacts["num_classes"] >= 2
    assert pre.scaler is not None


@pytest.mark.nn
def test_nn_preprocessor_testset(dummy_configs, dummy_dataframe):
    global_cfg, pre_cfg = dummy_configs
    pre = NNPreprocessor(global_cfg, pre_cfg)

    # Fit first to obtain encoder/scaler
    pre.df = dummy_dataframe.copy()
    pre.combine_rare_labels().encode_labels().split_features().scale_features()

    X, y = pre.preprocess_test(dummy_dataframe.copy(), pre.scaler, pre.label_encoder)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]


@pytest.mark.nn
def test_nn_preprocessor_inference(dummy_configs, dummy_dataframe):
    global_cfg, pre_cfg = dummy_configs
    pre = NNPreprocessor(global_cfg, pre_cfg)
    pre.df = dummy_dataframe.copy()
    pre.basic_preprocessing().split_features().scale_features()

    #print(dummy_dataframe, flush=True)

    df = dummy_dataframe.drop(columns=["Label"])
    X = pre.preprocess_inference(df, pre.scaler)
    assert isinstance(X, np.ndarray)
    assert X.shape[1] == 3


@pytest.mark.nn
def test_nn_preprocessor_single(dummy_configs, dummy_dataframe):
    global_cfg, pre_cfg = dummy_configs
    pre = NNPreprocessor(global_cfg, pre_cfg)
    pre.df = dummy_dataframe.copy()
    pre.basic_preprocessing().split_features().scale_features()

    sample = dummy_dataframe.drop(columns=["Label"]).iloc[0]
    X = pre.preprocess_single(sample, pre.scaler)
    assert X.shape[1] == 3



# -----------------------------
# TreePreprocessor tests
# -----------------------------
@patch("pandas.read_csv")
@pytest.mark.tree
def test_tree_preprocessor_pipeline(mock_read_csv, dummy_configs, dummy_dataframe):
    mock_read_csv.return_value = dummy_dataframe.copy()

    global_cfg, pre_cfg = dummy_configs
    pre = TreePreprocessor(global_cfg, pre_cfg)
    X_train, X_val, y_train, y_val, artifacts = pre.preprocess()

    assert isinstance(X_train, np.ndarray)
    assert len(y_train) > 0
    assert artifacts["num_classes"] >= 2


@pytest.mark.tree
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


@pytest.mark.tree
def test_tree_preprocessor_inference(dummy_configs, dummy_dataframe):
    global_cfg, pre_cfg = dummy_configs
    pre = TreePreprocessor(global_cfg, pre_cfg)
    X = pre.preprocess_inference(dummy_dataframe.copy())
    assert isinstance(X, np.ndarray)
    assert X.shape[1] == 3


@pytest.mark.tree
def test_tree_preprocessor_single(dummy_configs, dummy_dataframe):
    global_cfg, pre_cfg = dummy_configs
    pre = TreePreprocessor(global_cfg, pre_cfg)
    sample = dummy_dataframe.drop(columns=["Label"]).iloc[0]
    X = pre.preprocess_single(sample)
    assert X.shape[1] == 3


# -----------------------------
# Factory tests
# -----------------------------
@pytest.mark.factory
def test_factory_returns_nnpreprocessor(dummy_configs):
    global_cfg, pre_cfg = dummy_configs
    pre = PreprocessorFactory.get_preprocessor("nn", global_cfg, pre_cfg)
    assert isinstance(pre, NNPreprocessor)
    print("Returned NNPreprocessor successfully", flush=True)


@pytest.mark.factory
@pytest.mark.parametrize("model_type", ["tree", "rf"])
def test_factory_returns_treepreprocessor(dummy_configs, model_type):
    global_cfg, pre_cfg = dummy_configs
    pre = PreprocessorFactory.get_preprocessor(model_type, global_cfg, pre_cfg)
    assert isinstance(pre, TreePreprocessor)
    print(f"Returned TreePreprocessor for model_type={model_type}", flush=True)


"""
@pytest.mark.factory
def test_factory_unknown_model_type_raises(dummy_configs):
    global_cfg, pre_cfg = dummy_configs
    with pytest.raises(ValueError, match="Unknown model_type"):
        PreprocessorFactory.get_preprocessor("unknown_model", global_cfg, pre_cfg)
    print("Correctly raised ValueError for unknown model_type", flush=True)
"""





#pytest -v -s
#pytest -v -m nn
#pytest -v -m tree
