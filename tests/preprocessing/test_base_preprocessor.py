import pytest
from types import SimpleNamespace

from src.preprocessors.base import BasePreprocessor


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
# BasePreprocessor structure
# -----------------------------
def test_base_preprocessor_abstract_methods(dummy_configs):
    global_cfg, pre_cfg = dummy_configs
    with pytest.raises(TypeError):
        _ = BasePreprocessor(global_cfg, pre_cfg)
