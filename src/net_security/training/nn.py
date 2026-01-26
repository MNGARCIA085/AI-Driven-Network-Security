import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .base import BaseTrainer
from .callbacks import EarlyStopping,LRReducer
from net_security.models.nnet import NNModel
from net_security.utils.results import Results, Metrics
from net_security.evaluation.base import Evaluator


class NNTrainer(BaseTrainer):
    def __init__(self, num_classes, average):
        super().__init__(num_classes, average)
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # is it important for trees????
        self.batch_size = 32



    # --- Data loaders ---
    def create_loaders(self, X_train, y_train, X_val, y_val, batch_size):
        """ loaders for train and val"""
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.long)),
            batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                          torch.tensor(y_val, dtype=torch.long)),
            batch_size=self.batch_size, shuffle=False
        )
        return train_loader, val_loader



    # --- Training one epoch ---
    def train_one_epoch(self, model, loader, optimizer, criterion):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Collect predictions for training accuracy
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

        avg_loss = total_loss / len(loader)
        train_acc = (np.array(all_preds) == np.array(all_labels)).mean()

        return avg_loss, train_acc



    # --- Evaluation ---
    def eval_one_epoch(self, model, loader, criterion, return_raw=False):
        model.eval()
        preds, labels, probs = [], [], []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = model(xb)
                batch_size = xb.size(0)

                # Loss
                loss = criterion(out, yb)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Predictions
                prob = nn.functional.softmax(out, dim=1)
                probs.extend(prob.cpu().numpy())
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(yb.cpu().numpy())


        
        evaluator = Evaluator(self.average)
        metrics = evaluator.compute_metrics(
                labels,
                preds,
                total_loss,
                total_samples,
            )

        return metrics



    # --- Train a model ---
    def train(self, X_train, y_train, X_val, y_val, config) -> Results:
        """
        config like:
        config = {
            "model": {
                "hidden_dims": [128, 64],
                "dropout": 0.2,
            },
            "training": {
                "epochs": 50,
                "batch_size": 32,
                "lr": 1e-3,
            },
        }

        in trees i might not have training at all

        """

        # input size
        input_size = X_train.shape[1]

        # get configs
        model_config = config.get("model", {})
        train_config = config.get("training", {})


        # model
        model = NNModel(
            input_size=input_size,
            num_classes=self.num_classes,
            hidden1=model_config.get("hidden1"),
            hidden2=model_config.get("hidden2")
        ).to(self.device)

        train_loader, val_loader = self.create_loaders(X_train, y_train, X_val, y_val, train_config.get("batch_size"))
        optimizer = optim.Adam(model.parameters(), lr=train_config["lr"])
        criterion = nn.CrossEntropyLoss()


        # learning rate scheduler
        lr_scheduler = LRReducer(optimizer, mode="max", factor=0.5, patience=3)

        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        best_val_metric = -np.inf
        best_model_state = None
        early_stopping = EarlyStopping(patience=3, mode="max")  

        
        epochs = train_config.get('epochs')
        for epoch in range(epochs):
            # --- Training ---
            train_loss, train_acc = self.train_one_epoch(model, train_loader, optimizer, criterion)

            # --- Validation ---
            results_val = self.eval_one_epoch(model, val_loader, criterion)
            val_f1 = results_val["f1"]

            # --- for logging ---
            train_losses.append(train_loss)
            val_losses.append(results_val['avg_loss'])
            train_accs.append(train_acc)
            val_accs.append(results_val['accuracy'])

            # --- Scheduler step (optional) ---
            lr_scheduler.step(results_val["f1"])


            # --- Progress print ---
            print(
                f"[Epoch {epoch+1:03d}/{epochs}] "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={results_val['avg_loss']:.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"val_acc={results_val['accuracy']:.4f} | "
                f"val_f1={val_f1:.4f}"
            )


            # --- Check for best model ---
            if val_f1 > best_val_metric:
                best_val_metric = val_f1
                best_model_state = model.state_dict()  # save best weights

            # --- Early stopping check ---
            early_stop = -1
            early_stopping(val_f1) # monitor F1
            if early_stopping.stop:
                print(f"[EarlyStopping] Stopping at epoch {epoch+1}")
                early_stop = epoch + 1
                break

        # --- Load best model weights ---
        model.load_state_dict(best_model_state)


        # --- Return results ---
        return self._build_results(
            model=model,
            train_losses=train_losses,
            train_accs=train_accs,
            val_losses=val_losses,
            val_accs=val_accs,
            hyperparams={
                "lr":train_config.get("lr"),
                "final_lr" : optimizer.param_groups[0]['lr'],
                "hidden1": model_config.get("hidden1"),
                "hidden2": model_config.get("hidden2"),
                "batch_size": model_config.get("batch_size"),
                "optimizer": "Adam", # later -> tune it
                "early_stop": early_stop,
            }
        )




    










"""
for a diffrent lasdst batch

def train_one_epoch(self, model, loader, optimizer, criterion):
    
    Train the model for one epoch and return average loss per sample and training accuracy.
    
    Args:
        model: PyTorch model
        loader: DataLoader for training data
        optimizer: optimizer (e.g., Adam)
        criterion: loss function (e.g., CrossEntropyLoss)
    
    Returns:
        avg_loss: average loss per sample
        train_acc: training accuracy over the epoch
    
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_preds, all_labels = [], []

    for xb, yb in loader:
        xb, yb = xb.to(self.device), yb.to(self.device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size  # weighted by batch size
        total_samples += batch_size

        # Collect predictions for training accuracy
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

    avg_loss = total_loss / total_samples
    train_acc = (np.array(all_preds) == np.array(all_labels)).mean()

    return avg_loss, train_acc

"""


