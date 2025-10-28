import numpy as np
from src.utils.results import TestResults, Metrics


class BaseEvaluator:
    def __init__(self, cfg, model):
        # cfg -> eval config
        self.model = model
        self.average = cfg.metrics.average

    
    def evaluate(self, X, y) -> TestResults:
        """Evaluate on labeled data"""
        raise NotImplementedError


    def predict(self, X):
        """see later if i need it"""
        raise NotImplementedError


    def _build_results(self, metrics, preds, probs, labels) -> TestResults:
        return TestResults(
            metrics = Metrics(**metrics),
            preds = np.array(preds),
            probs = np.array(probs),
            labels = np.array(labels)
        )


