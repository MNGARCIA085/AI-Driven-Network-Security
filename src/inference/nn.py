# inference/nn_predictor.py
import torch
import numpy as np
from .base import BasePredictor


import torch.nn as nn



class NNPredictor(BasePredictor):
    """Predictor for multiclass neural network models."""

    def __init__(self, model, encoder, device="cpu"):
        super().__init__(model, encoder)
        self.device = device
        self.model.to(device)
        self.model.eval()


    def _to_tensor(self, X):
        X = np.array(X).reshape(1, -1) if np.ndim(X) == 1 else np.array(X)
        return torch.tensor(X, dtype=torch.float32).to(self.device)


    def predict_proba(self, X):
        """Return softmax probabilities for each class."""
        X_tensor = self._to_tensor(X)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


    def predict(self, X):
        """ Returns class indices """
        return np.argmax(self.predict_proba(X), axis=1)


    def predict_labels(self, X):
        # Return decoded class labels
        preds = self.predict(X)
        decoded = self.encoder.inverse_transform(preds)
        return decoded # gives DDODS, BENING....


    def predict_logits(self, X):
        """ raw preds"""
        X_tensor = self._to_tensor(X)
        with torch.no_grad():
            logits = self.model(X_tensor)
        return logits
    



    #.........
    """
    def predict_with_targets(self, loader):
        preds, labels, probs = [], [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = self.model(xb)
                prob = nn.functional.softmax(out, dim=1)
                probs.extend(prob.cpu().numpy())
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(yb.cpu().numpy())
        return np.array(preds), np.array(labels), np.array(probs)
    """





