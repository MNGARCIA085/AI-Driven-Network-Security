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

    @abstractmethod
    def predict_labels(self, X):
        """ Predict labels in a human readable way (DDoS)...."""
        pass




# predict with labels could be something that helps me later, but is not really needed    
