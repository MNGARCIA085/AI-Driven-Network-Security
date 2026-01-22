import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder
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


#--------------------------Preprocessing-----------------------------------#

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



#--------------------------Training-----------------------------------#
@pytest.fixture
def dummy_data():
    X_train = np.random.rand(20, 5)
    y_train = np.random.randint(0, 2, 20)
    X_val = np.random.rand(5, 5)
    y_val = np.random.randint(0, 2, 5)
    return X_train, y_train, X_val, y_val



# see if i use it
@pytest.fixture
def data_config_dict():
    return {
        "path": "dummy_path",
        "features": None,
        "batch_size": 8,
        "balance_factor": 1.0,
        "val_size": 0.2,
        "random_state": 42,
        "scaler_type": "none",
    }




#--------------------------Tuning-----------------------------------#
@pytest.fixture
def tree_tuning_cfg():
    class CFG:
        criterion = ["gini", "entropy"]
        max_depth = type("Range", (), {"min":1, "max":5})()
        min_samples_split = type("Range", (), {"min":2, "max":4})()
        num_samples = 1
        average = "binary"
    return CFG

@pytest.fixture
def nn_tuning_cfg():
    class CFG:
        hidden1 = [4, 8]
        hidden2 = [2, 4]
        batch_size = [2, 4]
        lr = type("Range", (), {"min": 1e-3, "max": 1e-2})()
        num_samples = 1
        average = "binary"
    return CFG



#--------------------------Inference-----------------------------------#
@pytest.fixture
def dummy_encoder():
    le = LabelEncoder()
    le.fit(["benign", "ddos"])
    return le


@pytest.fixture
def dummy_X():
    return np.array([[0.1, 0.2, 0.3],
                     [0.5, 0.4, 0.3]])

@pytest.fixture
def dummy_y():
    return np.array([0, 1])



#--------------------------Evaluation-----------------------------------#
@pytest.fixture
def y_true():
    return np.array([0, 1, 1, 0, 1])

@pytest.fixture
def y_pred():
    return np.array([0, 1, 0, 0, 1])