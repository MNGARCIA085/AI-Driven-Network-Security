

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.nnet import nn_model


from .base_trainer import BaseTrainer

import torch
from torch.utils.data import DataLoader, TensorDataset
from src.training.base_trainer import BaseTrainer
from src.models.nnet import nn_model  # your PyTorch model




import torch.nn as nn
import torch.optim as optim



class NNTrainer(BaseTrainer):
    def __init__(self, cfg, model_cfg, input_size, num_classes):
        super().__init__(cfg, model_cfg, input_size, num_classes)  # call BaseTrainer init if needed
        self.model = self._create_model()  # instantiate model here

    def _create_model(self):
        return nn_model(self.input_size, self.num_classes)

    def _to_tensorv0(self, X, y, target_dtype=torch.long):
        """Convert numpy arrays to torch tensors"""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=target_dtype)
        return X_tensor, y_tensor



    def _to_tensor(self, *args):
        """
        Flexible tensor conversion:
        - _to_tensor(X) → returns X_tensor
        - _to_tensor(X, y) → returns (X_tensor, y_tensor)
        """
        tensors = []
        for i, arg in enumerate(args):
            dtype = torch.float32 if i == 0 else torch.long
            tensors.append(torch.tensor(arg, dtype=dtype))
        return tensors if len(tensors) > 1 else tensors[0]


    def train(self, X_train, y_train, X_val=None, y_val=None):
        # convert to tensors
        X_train_tensor, y_train_tensor = self._to_tensor(X_train, y_train)
        X_val_tensor, y_val_tensor = self._to_tensor(X_val, y_val)

        # datasets and loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # model, loss, optimizer
        model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 2
        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()

            val_acc = correct / total
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_acc:.4f}")




    def predict(self, X):
        """Return numpy predictions (class indices) for given data."""
        self.model.eval()
        X_tensor = self._to_tensor(X)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()  # <--- return predictions as numpy array


    def evaluate(self, X_val, y_val):
        """Evaluate model on validation data"""
        self.model.eval()
        X_val_tensor, y_val_tensor = self._to_tensor(X_val, y_val)

        with torch.no_grad():
            outputs = self.model(X_val_tensor)
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == y_val_tensor).float().mean().item()

        print(f"Evaluation Accuracy: {acc:.4f}")
        return acc




# -------------------------
# Example usage
# -------------------------
# After training:
# y_pred = trainer.predict(X_val)
# Then you can safely do:
# from sklearn.metrics import accuracy_score, f1_score
# acc = accuracy_score(y_val, y_pred)
# f1 = f1_score(y_val, y_pred, average='macro')




"""
to log with parent run
import mlflow
import torch
from datetime import datetime

def run_experiments(models, train_fn):
    exp_name = f"compare_models_{datetime.now():%Y%m%d_%H%M%S}"
    mlflow.set_experiment(exp_name)

    results = []

    with mlflow.start_run(run_name="comparison") as parent_run:
        for name, model, params in models:
            with mlflow.start_run(run_name=name, nested=True):
                metrics = train_fn(model, params)
                mlflow.log_metrics(metrics)
                mlflow.log_params(params)
                results.append((name, metrics, model))

        # choose best
        best = min(results, key=lambda x: x[1]["val_loss"])
        best_name, best_metrics, best_model = best

        # log summary in parent run
        mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})
        mlflow.log_param("best_model", best_name)

        # save only the best
        model_path = f"best_model_{best_name}.pt"
        torch.save(best_model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
"""