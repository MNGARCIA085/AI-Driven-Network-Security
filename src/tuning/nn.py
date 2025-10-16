import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import ray
from src.models.nnet import NNModel
import numpy as np


class NNTuner:
    def __init__(self, cfg, X_train, y_train, X_val, y_val, num_classes, average="weighted"):
        self.cfg = cfg
        self.X_train_id = ray.put(X_train)
        self.y_train_id = ray.put(y_train)
        self.X_val_id = ray.put(X_val)
        self.y_val_id = ray.put(y_val)
        self.input_size = X_train.shape[1]
        self.num_classes = num_classes
        self.average = average

        # device once
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data loaders ---
    def create_loaders(self, batch_size):
        X_train = ray.get(self.X_train_id)
        y_train = ray.get(self.y_train_id)
        X_val = ray.get(self.X_val_id)
        y_val = ray.get(self.y_val_id)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.long)),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                          torch.tensor(y_val, dtype=torch.long)),
            batch_size=batch_size, shuffle=False
        )
        return train_loader, val_loader

    # --- Training one epoch ---
    def train_one_epoch(self, model, loader, optimizer, criterion):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    # --- Evaluation ---
    def eval_one_epoch(self, model, loader, criterion):
        model.eval()
        preds, labels, probs = [], [], []
        total_loss = 0.0

        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = model(xb)
                total_loss += criterion(out, yb).item()
                prob = nn.functional.softmax(out, dim=1)
                probs.extend(prob.cpu().numpy())
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(yb.cpu().numpy())

        avg_loss = total_loss / len(loader)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average=self.average, zero_division=0)
        return f1, avg_loss, acc, preds, labels, np.array(probs)



    # --- Ray train function ---
    def _train_model_ray(self, config):
        model = NNModel(
            input_size=self.input_size,
            num_classes=self.num_classes,
            hidden1=config["hidden1"],
            hidden2=config["hidden2"]
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.CrossEntropyLoss()
        train_loader, val_loader = self.create_loaders(config["batch_size"])

        for _ in range(self.cfg.tuning.epochs_trials):  # epochs for tuning
            self.train_one_epoch(model, train_loader, optimizer, criterion)
            f1, _, _, _, _,_ = self.eval_one_epoch(model, val_loader, criterion)
            tune.report({"f1": f1})

    # --- Tune hyperparameters ---
    def tune(self, num_samples=5):

        config = {
            "hidden1": tune.choice(self.cfg.tuning.hidden1),
            "hidden2": tune.choice(self.cfg.tuning.hidden2),
            "batch_size": tune.choice(self.cfg.tuning.batch_size),
            "lr": tune.loguniform(self.cfg.tuning.lr.min, self.cfg.tuning.lr.max),
        }

        scheduler = ASHAScheduler(metric="f1", mode="max")
        tuner = tune.Tuner(
            tune.with_parameters(self._train_model_ray),
            param_space=config,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=1, #num_samples
            )
        )

        results = tuner.fit()
        best = results.get_best_result(metric="f1", mode="max")
        return best.config

    # --- Train best model ---
    def train_best_model(self, config):
        model = NNModel(
            input_size=self.input_size,
            num_classes=self.num_classes,
            hidden1=config["hidden1"],
            hidden2=config["hidden2"]
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.CrossEntropyLoss()
        train_loader, val_loader = self.create_loaders(config["batch_size"])

        train_losses, val_losses, val_accs = [], [], []
        all_val_preds, all_val_labels = [], []

        for _ in range(self.cfg.tuning.epochs):
            train_loss = self.train_one_epoch(model, train_loader, optimizer, criterion)
            f1, val_loss, val_acc, preds, labels, probs = self.eval_one_epoch(model, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
        
        # final epoch preds.
        # 2️⃣ Make predictions with the trained model
        all_val_preds, all_val_labels, all_val_probs = self.predict(val_loader, model)

        metrics = {
            "accuracy": accuracy_score(all_val_labels, all_val_preds),
            "precision": precision_score(all_val_labels, all_val_preds, average=self.average, zero_division=0),
            "recall": recall_score(all_val_labels, all_val_preds, average=self.average, zero_division=0),
            "f1": f1_score(all_val_labels, all_val_preds, average=self.average, zero_division=0),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "val_preds": all_val_preds,
            "val_labels": all_val_labels,
            "model": model,
            "val_preds_proba": all_val_probs,  # added for ROC
        }
        return metrics


    def predict(self, loader, model):
        model.eval()
        preds, labels, probs = [], [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = model(xb)
                prob = nn.functional.softmax(out, dim=1)
                probs.extend(prob.cpu().numpy())
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(yb.cpu().numpy())
        return np.array(preds), np.array(labels), np.array(probs)
