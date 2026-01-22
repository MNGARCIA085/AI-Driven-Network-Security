from src.inference.nn import NNPredictor
import torch
import numpy as np


class DummyNN(torch.nn.Module):
    def __init__(self, input_size=3, num_classes=2):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)



def test_nn_predictor_single_and_batch(dummy_X, dummy_encoder):
    model = DummyNN(input_size=3, num_classes=2)
    predictor = NNPredictor(model, dummy_encoder, device="cpu")

    # Single sample
    pred_single = predictor.predict(dummy_X[0])
    assert pred_single.shape[0] == 1

    # Batch
    pred_batch = predictor.predict(dummy_X)
    assert pred_batch.shape[0] == dummy_X.shape[0]

    # Predict probabilities
    probs = predictor.predict_proba(dummy_X)
    assert probs.shape == (dummy_X.shape[0], 2)
    np.testing.assert_almost_equal(probs.sum(axis=1), np.ones(dummy_X.shape[0]))

    # Predict labels
    labels = predictor.predict_labels(dummy_X)
    assert labels.shape[0] == dummy_X.shape[0]
    assert set(labels).issubset({"benign", "ddos"})