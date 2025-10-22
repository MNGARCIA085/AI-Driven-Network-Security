from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import ray
from .base import BaseTuner
from src.utils.metrics import compute_metrics



class TreeTuner(BaseTuner):
    def __init__(self, cfg, X_train, y_train, X_val, y_val):
        super().__init__(cfg, X_train, y_train, X_val, y_val)


    # --- Evaluation ---
    def eval_model(self, model, X, y):
        preds = model.predict(X)
        probs = model.predict_proba(X)  # for ROC and AUC
        metrics = compute_metrics(y, preds, self.average)
        return {**metrics, "preds": np.array(preds), "labels": np.array(y), "probs": np.array(probs)}


    # --- Ray train function ---
    def _train_model_ray(self, config):
        X_train, y_train = ray.get(self.X_train_id), ray.get(self.y_train_id)
        X_val, y_val = ray.get(self.X_val_id), ray.get(self.y_val_id)

        model = DecisionTreeClassifier(
            criterion=config["criterion"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            random_state=42, #self.cfg["random_state"]
        ) # defined in other place????????

        model.fit(X_train, y_train)
        results_val = self.eval_model(model, X_val, y_val)

        # report to tune
        tune.report({"f1": results_val['f1']})



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

        results_val = self.eval_model(model, X_val, y_val)
        results_val["val_labels"] = results_val.pop("labels")
        results_val["val_preds"] = results_val.pop("preds")
        results_val["val_preds_proba"] = results_val.pop("probs")
        
        return {
            "model": model,
            **results_val
        }




