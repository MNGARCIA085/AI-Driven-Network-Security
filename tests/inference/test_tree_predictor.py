import pytest
import numpy as np
from unittest.mock import MagicMock
from src.inference.tree import TreePredictor


@pytest.fixture
def mock_components():
    model = MagicMock()
    encoder = MagicMock()
    return model, encoder


@pytest.fixture
def predictor(mock_components):
    model, encoder = mock_components
    return TreePredictor(model, encoder)


def test_predict_proba_returns_numpy_array(predictor, mock_components):
    model, _ = mock_components
    mock_probs = np.array([[0.2, 0.8]])
    model.predict_proba.return_value = mock_probs

    X = [[1, 2, 3]]
    probs = predictor.predict_proba(X)

    model.predict_proba.assert_called_once()
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (1, 2)
    np.testing.assert_array_equal(probs, mock_probs)


def test_predict_decodes_labels(predictor, mock_components):
    model, encoder = mock_components
    mock_preds = np.array([1])
    model.predict.return_value = mock_preds
    encoder.inverse_transform.return_value = ["classB"]

    X = [[1, 2, 3]]
    decoded = predictor.predict(X)

    model.predict.assert_called_once()
    encoder.inverse_transform.assert_called_once_with(mock_preds)
    assert decoded == ["classB"]


def test_predict_single_sample(predictor, mock_components):
    model, encoder = mock_components
    mock_preds = np.array([0])
    model.predict.return_value = mock_preds
    encoder.inverse_transform.return_value = ["classA"]

    X = [1, 2, 3]  # single sample
    decoded = predictor.predict(X)

    model.predict.assert_called_once()
    assert decoded == ["classA"]


def test_input_reshaping_for_single_sample(predictor, mock_components):
    model, _ = mock_components
    mock_probs = np.array([[0.5, 0.5]])
    model.predict_proba.return_value = mock_probs

    X = [4, 5, 6]
    probs = predictor.predict_proba(X)

    assert probs.shape == (1, 2)


"""
predict_proba returns correct numpy array with proper shape

predict decodes labels correctly via encoder.inverse_transform

Handles single and batch input shapes

Ensures reshaping works for single samples
"""