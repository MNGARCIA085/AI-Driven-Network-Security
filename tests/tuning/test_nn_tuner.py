from unittest.mock import patch
from net_security.tuning.nn import NNTuner


def test_nn_tuner_init(dummy_data, nn_tuning_cfg):
    X_train, y_train, X_val, y_val = dummy_data
    tuner = NNTuner(nn_tuning_cfg, X_train, y_train, X_val, y_val, num_classes=2)
    
    assert tuner.num_classes == 2
    assert hasattr(tuner, "device")
    assert hasattr(tuner, "X_train_id")




@patch("ray.tune.Tuner.fit")
def test_nn_tuner_tune(mock_fit, dummy_data, nn_tuning_cfg):
    mock_fit.return_value.get_best_result.return_value.config = {"model.hidden1":4}
    X_train, y_train, X_val, y_val = dummy_data
    tuner = NNTuner(nn_tuning_cfg, X_train, y_train, X_val, y_val, num_classes=2)
    best_cfg = tuner.tune()
    
    assert best_cfg["model.hidden1"] == 4