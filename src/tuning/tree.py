from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np

import ray




from .base import BaseTuner



class TreeTuner(BaseTuner):
    def __init__(self, cfg, X_train, y_train, X_val, y_val):
        super().__init__(cfg, X_train, y_train, X_val, y_val)



    # --- Evaluation ---
    def eval_model(self, model, X, y):
        preds = model.predict(X)
        probs = model.predict_proba(X)  # for ROC and AUC
        metrics = {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, average=self.average, zero_division=0),
            "recall": recall_score(y, preds, average=self.average, zero_division=0),
            "f1": f1_score(y, preds, average=self.average, zero_division=0),
        }
        return metrics, preds, probs

    # --- Ray train function ---
    def _train_model_ray(self, config):
        X_train, y_train = ray.get(self.X_train_id), ray.get(self.y_train_id)
        X_val, y_val = ray.get(self.X_val_id), ray.get(self.y_val_id)

        model = DecisionTreeClassifier(
            criterion=config["criterion"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            random_state=42, #self.cfg["random_state"]
        )

        model.fit(X_train, y_train)
        metrics, _, _ = self.eval_model(model, X_val, y_val)


        f1 = metrics["f1"]
        tune.report({"f1": f1})

        

    # get tune config
    def get_tune_config(self):
        return {
            "criterion": tune.choice(self.cfg.criterion),
            "max_depth": tune.randint(self.cfg.max_depth.min, self.cfg.max_depth.max),
            "min_samples_split": tune.randint(self.cfg.min_samples_split.min, self.cfg.min_samples_split.max)
        }

    # --- Train best model ---
    def train_best_model(self, config):
        X_train, y_train = ray.get(self.X_train_id), ray.get(self.y_train_id)
        X_val, y_val = ray.get(self.X_val_id), ray.get(self.y_val_id)

        model = DecisionTreeClassifier(
            criterion=config["criterion"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            random_state=42,   #self.cfg["random_state"]
        )
        model.fit(X_train, y_train)

        metrics, val_preds, probs = self.eval_model(model, X_val, y_val)
        return {
            "model": model,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "val_preds": val_preds,
            "val_labels": y_val,
            "val_preds_proba": probs,
        }




