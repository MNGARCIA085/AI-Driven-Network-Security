from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np

import ray



class TreeTuner:
    def __init__(self, cfg, X_train, y_train, X_val, y_val, average="weighted"):
        self.cfg = cfg
        self.X_train_id = ray.put(X_train)
        self.y_train_id = ray.put(y_train)
        self.X_val_id = ray.put(X_val)
        self.y_val_id = ray.put(y_val)
        self.average = average

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

        #tune.report(f1=metrics["f1"])

    # --- Tune hyperparameters ---
    def tune(self, num_samples=10):
        

        """
        config = {
            "criterion": tune.choice(["gini", "entropy", "log_loss"]),
            "max_depth": tune.randint(2, 20),
            "min_samples_split": tune.randint(2, 10)
        }
        """

        config = {
            "criterion": tune.choice(self.cfg.criterion),
            "max_depth": tune.randint(self.cfg.max_depth.min, self.cfg.max_depth.max),
            "min_samples_split": tune.randint(self.cfg.min_samples_split.min, self.cfg.min_samples_split.max)
        }

        scheduler = ASHAScheduler(metric="f1", mode="max")
        
        tuner = tune.Tuner(
            tune.with_parameters(self._train_model_ray),
            param_space=config,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=2 #num_samples
            )
        )
        results = tuner.fit()
        best = results.get_best_result(metric="f1", mode="max")
        return best.config

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











# abstract class later to be coherent