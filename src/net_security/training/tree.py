from sklearn.tree import DecisionTreeClassifier
import numpy as np
from .base import BaseTrainer

from net_security.models.tree import TreeModel
from net_security.utils.results import Results



class TreeTrainer(BaseTrainer):
    def __init__(self,  num_classes, average):
        super().__init__(num_classes, average)


    # --- Train ---
    def train(self, X_train, y_train, X_val, y_val, config) -> Results:

        # get configs
        model_config = config.get("model", {})
        train_config = config.get("training", {})


        model = TreeModel(
            criterion=model_config.get("criterion"),
            max_depth=model_config.get("max_depth"),
            min_samples_split=model_config.get("min_samples_split"),
            random_state=42,
        )

        model.fit(X_train, y_train)


        # return results
        return self._build_results(
            model=model,
            hyperparams={
                "criterion":model_config.get("criterion"),
                "max_depth":model_config.get("max_depth"),
                "min_samples_split":model_config.get("min_samples_split"),
            }
        )






