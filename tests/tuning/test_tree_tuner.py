from unittest.mock import patch
from src.tuning.tree import TreeTuner

def test_tree_tuner_init(dummy_data, tree_tuning_cfg):
    X_train, y_train, X_val, y_val = dummy_data
    tuner = TreeTuner(tree_tuning_cfg, X_train, y_train, X_val, y_val, num_classes=2)
    
    assert tuner.num_classes == 2
    assert hasattr(tuner, "X_train_id")
    assert hasattr(tuner, "X_val_id")


# Mock Ray Tune to avoid long execution
@patch("ray.tune.Tuner.fit")
def test_tree_tuner_tune(mock_fit, dummy_data, tree_tuning_cfg):
    mock_fit.return_value.get_best_result.return_value.config = {"model.criterion":"gini"}
    
    X_train, y_train, X_val, y_val = dummy_data
    tuner = TreeTuner(tree_tuning_cfg, X_train, y_train, X_val, y_val, num_classes=2)
    best_cfg = tuner.tune()
    
    assert best_cfg["model.criterion"] == "gini"