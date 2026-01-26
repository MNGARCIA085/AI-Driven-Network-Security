from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import ray
from .base import BaseTuner


from net_security.models.tree import TreeModel
from net_security.utils.results import Results, Metrics
from net_security.inference.tree import TreePredictor
from net_security.evaluation.base import Evaluator


class TreeTuner(BaseTuner):
    def __init__(self, cfg, X_train, y_train, X_val, y_val, num_classes):
        super().__init__(cfg, X_train, y_train, X_val, y_val, num_classes) 

    # get tune config
    def get_tune_config(self):
        return {
            "model.criterion": tune.choice(self.cfg.criterion),
            "model.max_depth": tune.randint(self.cfg.max_depth.min, self.cfg.max_depth.max),
            "model.min_samples_split": tune.randint(self.cfg.min_samples_split.min, self.cfg.min_samples_split.max)
        }


    # --- Ray train function ---
    @staticmethod
    def train_model_ray(config, X_train, y_train, X_val, y_val, average, num_classes): 

        model = TreeModel(
            criterion=config["model.criterion"],
            max_depth=config["model.max_depth"],
            min_samples_split=config["model.min_samples_split"],
            random_state=42,
        )

        model.fit(X_train, y_train)

        # -- Preds and evaluation
        predictor = TreePredictor(model)
        preds = predictor.predict(X_val)

        # -- Eval
        evaluator = Evaluator(average) 
        metrics = evaluator.compute_metrics(y_val, preds)

        # report to tune
        tune.report({"f1": metrics['f1']})

