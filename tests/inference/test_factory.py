import pytest
from unittest.mock import MagicMock
from src.inference.factory import PredictorFactory
from src.inference.nn import NNPredictor
from src.inference.tree import TreePredictor


def test_get_predictor_nn_returns_nn_predictor():
    model = MagicMock()
    encoder = MagicMock()
    scaler = MagicMock()

    predictor = PredictorFactory.get_predictor("nn", model, encoder, scaler=scaler, device="cpu")

    assert isinstance(predictor, NNPredictor)
    assert predictor.model is model
    assert predictor.encoder is encoder
    assert predictor.scaler is scaler
    assert predictor.device == "cpu"


def test_get_predictor_tree_returns_tree_predictor():
    model = MagicMock()
    encoder = MagicMock()

    predictor = PredictorFactory.get_predictor("tree", model, encoder)

    assert isinstance(predictor, TreePredictor)
    assert predictor.model is model
    assert predictor.encoder is encoder


def test_get_predictor_invalid_type_raises():
    model = MagicMock()
    encoder = MagicMock()

    with pytest.raises(ValueError, match="Unknown tuner type: invalid"):
        PredictorFactory.get_predictor("invalid", model, encoder)
