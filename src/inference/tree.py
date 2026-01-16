import numpy as np
from .base import BasePredictor

class TreePredictor(BasePredictor):
    """Predictor for tree-based models (e.g., DecisionTree, RandomForest)."""

    def __init__(self, model, encoder):
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
        







"""
def eval_model(self, model, X, y):
        preds = model.predict(X)
        probs = model.predict_proba(X)  # for ROC and AUC
        metrics = compute_metrics(y, preds, self.average)
        return {"metrics":metrics, "preds": np.array(preds), "labels": np.array(y), "probs": np.array(probs)}
"""