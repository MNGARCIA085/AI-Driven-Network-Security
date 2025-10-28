from src.utils.metrics import compute_metrics
from .base import BaseEvaluator
from src.utils.results import TestResults



class TreeEvaluator(BaseEvaluator):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)

    def evaluate(self, X, y, probs=None) -> TestResults:
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        metrics = compute_metrics(y, preds)
        
        # return
        return self._build_results(
            metrics=metrics,
            preds=preds,
            probs=probs,
            labels=labels
        )


