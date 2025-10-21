import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


from .base import BaseEvaluator



class NNEvaluator(BaseEvaluator):
    def __init__(self, model, device="cpu", batch_size=32, average="macro"):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.average = average
        self.model.to(self.device)
        self.model.eval()




    def _to_loader(self, X, y=None):
        """Convert X (and optionally y) to a DataLoader."""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)



    def evaluate(self, X, y):
        """
        Evaluate on labeled data.
        Returns: f1, avg_loss, acc, preds, labels, probs
        """

        criterion = nn.CrossEntropyLoss() # check later how do I pass it


        loader = self._to_loader(X, y)
        preds, labels, probs = [], [], []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                xb = batch[0].to(self.device)
                yb = batch[1].to(self.device)
                out = self.model(xb)

                # Loss
                loss = criterion(out, yb)
                total_loss += loss.item() * xb.size(0)
                
                total_samples += xb.size(0)

                # Predictions
                prob = F.softmax(out, dim=1)
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(yb.cpu().numpy())
                probs.extend(prob.cpu().numpy())

        avg_loss = total_loss / total_samples if criterion is not None else None
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average=self.average, zero_division=0)

        return f1, avg_loss, acc, np.array(preds), np.array(labels), np.array(probs)






    """
    def predict(self, X):
        #Run inference on unlabeled data (batch or single sample).  -------------> for inference later
        loader = self._to_loader(X)
        preds, probs = [], []

        with torch.no_grad():
            for batch in loader:
                xb = batch[0].to(self.device)
                out = self.model(xb)
                prob = F.softmax(out, dim=1)
                preds.extend(out.argmax(1).cpu().numpy())
                probs.extend(prob.cpu().numpy())

        return np.array(preds), np.array(probs)



    def predict_single(self, x):
        #Predict a single example.
        self.model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x_tensor) # raw logits
            prob = F.softmax(out, dim=1).cpu().numpy()[0] # probs.
            pred = out.argmax(1).item() # predicted label (for ex. 7)
            # pred_class = torch.argmax(probs, dim=1)
        return pred, prob
    """






