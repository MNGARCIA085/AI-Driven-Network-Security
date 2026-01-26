from sklearn.tree import DecisionTreeClassifier
import numpy as np
from net_security.inference.tree import TreePredictor


def test_tree_predictor_single_and_batch(dummy_X, dummy_y, dummy_encoder):
    model = DecisionTreeClassifier()
    model.fit(dummy_X, dummy_y)
    predictor = TreePredictor(model, encoder=dummy_encoder)

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
