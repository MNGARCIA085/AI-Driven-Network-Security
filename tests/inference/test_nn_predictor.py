import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from src.inference.nn import NNPredictor


@pytest.fixture
def mock_components():
    model = MagicMock()
    encoder = MagicMock()
    scaler = MagicMock()
    return model, encoder, scaler


@pytest.fixture
def predictor(mock_components):
    model, encoder, scaler = mock_components
    return NNPredictor(model, encoder, scaler, device="cpu")


def test_to_tensor_scaling_and_shape(predictor, mock_components):
    _, _, scaler = mock_components
    scaler.transform.return_value = np.array([[1.0, 2.0, 3.0]])
    X = [0.5, 0.2, 0.1]

    tensor = predictor._to_tensor(X)

    scaler.transform.assert_called_once()
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3)
    assert tensor.dtype == torch.float32


def test_predict_proba_returns_numpy_array(predictor, mock_components):
    model, _, scaler = mock_components
    scaler.transform.return_value = np.array([[0.1, 0.2]])
    mock_logits = torch.tensor([[1.0, 2.0, 3.0]])
    model.return_value = mock_logits

    X = [[0.1, 0.2]]
    probs = predictor.predict_proba(X)

    model.assert_called_once()
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (1, 3)
    np.testing.assert_almost_equal(probs.sum(), 1.0, decimal=5)  # softmax sums to 1


def test_predict_decodes_labels(predictor, mock_components):
    model, encoder, scaler = mock_components
    scaler.transform.return_value = np.array([[0.1, 0.2]])
    mock_logits = torch.tensor([[1.0, 2.0, 3.0]])
    model.return_value = mock_logits
    encoder.inverse_transform.return_value = ["class2"]

    X = [[0.1, 0.2]]
    preds = predictor.predict(X)

    encoder.inverse_transform.assert_called_once()
    assert preds == ["class2"]


def test_predict_single_sample_shape(predictor, mock_components):
    model, encoder, scaler = mock_components
    scaler.transform.return_value = np.array([[0.1, 0.2]])
    model.return_value = torch.tensor([[0.0, 1.0]])
    encoder.inverse_transform.return_value = ["class1"]

    X = [0.1, 0.2]  # single sample
    preds = predictor.predict(X)

    assert preds == ["class1"]


"""
Tensor conversion and scaling (_to_tensor)

Proper handling of single and batch inputs

Output types and shapes (predict_proba → numpy array)

Decoding logic (predict)

That softmax probabilities sum to 1
"""