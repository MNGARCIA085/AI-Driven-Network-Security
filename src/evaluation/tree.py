


from src.utils.metrics import compute_metrics
from .base import BaseEvaluator
import numpy as np

class TreeEvaluator(BaseEvaluator):
    def __init__(self, model, average="weighted"):
        self.model = model
        self.average = average

    def evaluate(self, X, y, probs=None):
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)  # for ROC and AUC
        metrics = compute_metrics(y, preds)
        return {**metrics, "preds": np.array(preds), "labels": np.array(y), "probs": np.array(probs)}


