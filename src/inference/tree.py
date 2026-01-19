import numpy as np
from .base import BasePredictor

class TreePredictor(BasePredictor):
    """Predictor for tree-based models (e.g., DecisionTree, RandomForest)."""

    def __init__(self, model, encoder=None):
        super().__init__(model, encoder)

    def predict_proba(self, X):
        """Return class probabilities."""
        X = np.array(X).reshape(1, -1) if np.ndim(X) == 1 else np.array(X)
        return self.model.predict_proba(X)


    def predict(self, X): # -> already gives class indices
        """Return raw preds."""
        X = np.array(X).reshape(1, -1) if np.ndim(X) == 1 else np.array(X)
        return self.model.predict(X)



    def predict_labels(self, X):
        """Return decoded class labels."""
        preds = self.predict(X)
        decoded = self.encoder.inverse_transform(preds)
        return decoded #return decoded[0] if np.ndim(X) == 1 else decoded
        

