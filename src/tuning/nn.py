import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np

class NNModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden1, hidden2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.model(x)


import ray


class NNTuner:
    def __init__(self, X_train, y_train, X_val, y_val, num_classes, average="macro"):
        self.X_train_id = ray.put(X_train)
        self.y_train_id = ray.put(y_train)
        self.X_val_id = ray.put(X_val)
        self.y_val_id = ray.put(y_val)
        self.num_classes = num_classes
        self.average = average

    def _train_model_ray(self, config):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        from ray import tune
        from sklearn.metrics import f1_score

        # retrieve large arrays from Ray object store
        X_train = ray.get(self.X_train_id)
        y_train = ray.get(self.y_train_id)
        X_val = ray.get(self.X_val_id)
        y_val = ray.get(self.y_val_id)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = NNModel(
            input_size=X_train.shape[1],
            num_classes=self.num_classes,
            hidden1=config["hidden1"],
            hidden2=config["hidden2"]
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.long)),
            batch_size=config["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                          torch.tensor(y_val, dtype=torch.long)),
            batch_size=256, shuffle=False
        )

        for _ in range(2): #10
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            # validation
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    out = model(xb)
                    preds.extend(out.argmax(1).cpu().numpy())
                    labels.extend(yb.numpy())

            f1 = f1_score(labels, preds, average=self.average)
            #tune.report(f1=f1)
            tune.report({"f1": f1})





    def tune(self, num_samples=5):
        config = {
            "hidden1": tune.choice([64, 128, 256]),
            "hidden2": tune.choice([32, 64, 128]),
            "batch_size": tune.choice([32, 64]),
            "lr": tune.loguniform(1e-4, 1e-2),
        }

        scheduler = ASHAScheduler(metric="f1", mode="max")
        tuner = tune.Tuner(
            tune.with_parameters(self._train_model_ray),
            param_space=config,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=1 #num_samples,
                #metric="f1",
                #mode="max"
            )
        )

        results = tuner.fit()
        best = results.get_best_result(metric="f1", mode="max")
        return best.config

    # === Train the best model fully and return all relevant info ===
    def train_best_model(self, config, epochs=5):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # retrieve the arrays from Ray object store
        X_train = ray.get(self.X_train_id)
        y_train = ray.get(self.y_train_id)
        X_val = ray.get(self.X_val_id)
        y_val = ray.get(self.y_val_id)

        model = NNModel(
            input_size=X_train.shape[1],
            num_classes=self.num_classes,
            hidden1=config["hidden1"],
            hidden2=config["hidden2"]
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.long)),
            batch_size=config["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                          torch.tensor(y_val, dtype=torch.long)),
            batch_size=256, shuffle=False
        )

        train_losses, val_losses, val_accs = [], [], []
        all_val_preds, all_val_labels = [], []

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_losses.append(total_loss / len(train_loader))

            # validation
            model.eval()
            val_loss = 0
            preds, labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    val_loss += criterion(out, yb).item()
                    pred_labels = out.argmax(1).cpu().numpy()
                    preds.extend(pred_labels)
                    labels.extend(yb.cpu().numpy())

            val_loss /= len(val_loader)
            acc = accuracy_score(labels, preds)
            val_accs.append(acc)
            val_losses.append(val_loss)
            all_val_preds = preds
            all_val_labels = labels

        # final metrics
        metrics = {
            "accuracy": accuracy_score(all_val_labels, all_val_preds),
            "precision": precision_score(all_val_labels, all_val_preds, average=self.average, zero_division=0),
            "recall": recall_score(all_val_labels, all_val_preds, average=self.average, zero_division=0),
            "f1": f1_score(all_val_labels, all_val_preds, average=self.average, zero_division=0),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accs": val_accs,
             #"val_preds": all_val_preds,
             #"val_labels": all_val_labels,
            "model": model
        }

        return metrics
