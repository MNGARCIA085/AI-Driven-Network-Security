import pandas as pd
import numpy as np
import pytest
from src.config.data import DataConfig



# reminder -> not hardcode Label in preprocessor!!


"""
@pytest.fixture
def small_df():
    return pd.DataFrame({
        "f1": [1,2,3,4,5,6,7,8],
        "f2": [10,20,30,40,50,60,70,80],
        "Label": ["normal","attack"] * 4,
    })
"""

@pytest.fixture
def small_df():
    df = pd.DataFrame({
        "f1": np.arange(20),
        "f2": np.arange(20, 40),
        "Label": ["attack"] * 10 + ["normal"] * 10,
    })
    return df





@pytest.fixture
def global_cfg():
    return {
        "batch_size": 64,
        "random_state": 42,
    }

@pytest.fixture
def data_cfg_nn(tmp_path):
    return DataConfig(
        path=str(tmp_path),
        features=None,
        batch_size=64,
        balance_factor=1.0,
        val_size=0.25,
        random_state=42,
        scaler_type="standard",
    )

@pytest.fixture
def data_cfg_tree(tmp_path):
    return DataConfig(
        path=str(tmp_path),
        features=None,
        batch_size=64,
        balance_factor=1.0,
        val_size=0.25,
        random_state=42,
        scaler_type="none",
    )
