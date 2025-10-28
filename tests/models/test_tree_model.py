# test_tree_model.py
import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.models.tree import TreeModel


def test_tree_model_basic():
    # Sample data
    X = np.random.rand(10, 4)
    y = np.random.randint(0, 3, 10)  # multiclass
    
    # Create model
    model = TreeModel(
        criterion="gini",
        max_depth=3,
        min_samples_split=2,
        random_state=42
    )
    
    # Check type
    assert isinstance(model, DecisionTreeClassifier)
    
    # Fit
    model.fit(X, y)
    
    # Predict
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]
    
    # Predict_proba
    probs = model.predict_proba(X)
    assert probs.shape == (X.shape[0], 3)
    assert (probs >= 0).all() and (probs <= 1).all()
