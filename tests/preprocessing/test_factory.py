import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace

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



# -----------------------------
# Factory tests
# -----------------------------
def test_factory_returns_nnpreprocessor(dummy_configs):
    global_cfg, pre_cfg = dummy_configs
    pre = PreprocessorFactory.get_preprocessor("nn", global_cfg, pre_cfg)
    assert isinstance(pre, NNPreprocessor)
    print("Returned NNPreprocessor successfully", flush=True)



@pytest.mark.parametrize("model_type", ["tree", "rf"])
def test_factory_returns_treepreprocessor(dummy_configs, model_type):
    global_cfg, pre_cfg = dummy_configs
    pre = PreprocessorFactory.get_preprocessor(model_type, global_cfg, pre_cfg)
    assert isinstance(pre, TreePreprocessor)
    print(f"Returned TreePreprocessor for model_type={model_type}", flush=True)



def test_factory_unknown_model_type_raises(dummy_configs):
    global_cfg, pre_cfg = dummy_configs
    with pytest.raises(ValueError, match="Unknown model_type"):
        PreprocessorFactory.get_preprocessor("unknown_model", global_cfg, pre_cfg)
    print("Correctly raised ValueError for unknown model_type", flush=True)


#pytest -v -s