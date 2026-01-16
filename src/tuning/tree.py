from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import ray
from .base import BaseTuner
from src.utils.metrics import compute_metrics
from src.models.tree import TreeModel
from src.utils.results import Results, Metrics



class TreeTuner(BaseTuner):
    def __init__(self, cfg, X_train, y_train, X_val, y_val, num_classes):
        super().__init__(cfg, X_train, y_train, X_val, y_val) 

        self.num_classes = num_classes # maybe to parent later!!!!





    # --- Ray train function ---
    @staticmethod
    def train_model_ray(config, X_train_id, y_train_id, X_val_id, y_val_id, num_classes):

        X_train = X_train_id
        y_train = y_train_id
        X_val = X_val_id
        y_val = y_val_id

        model = TreeModel(
            criterion=config["criterion"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            random_state=42,
        )

        model.fit(X_train, y_train)

        # -- juts for now, later i can be better, maybe predictor and eval class
        preds = model.predict(X_val)
        metrics = compute_metrics(y_val, preds, 'weighted') # change avg. later; pass as a vars!!!!



        # report to tune
        tune.report({"f1": metrics['f1']})



    # get tune config
    def get_tune_config(self):
        return {
            "criterion": tune.choice(self.cfg.criterion),
            "max_depth": tune.randint(self.cfg.max_depth.min, self.cfg.max_depth.max),
            "min_samples_split": tune.randint(self.cfg.min_samples_split.min, self.cfg.min_samples_split.max)
        }

    












