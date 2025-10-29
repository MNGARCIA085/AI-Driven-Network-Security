# inference/base_predictor.py
from abc import ABC, abstractmethod

class BasePredictor(ABC):
    """Abstract base class for predictors."""

    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder

    @abstractmethod
    def predict(self, X):
        """Predict labels for a single sample or a batch."""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Predict probabilities for a single sample or a batch (if supported)."""
        pass
