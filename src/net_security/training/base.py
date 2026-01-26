import ray
from abc import ABC, abstractmethod
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from net_security.utils.results import Results, Metrics



class BaseTrainer():
    """
    Base class for training models.

    Attributes:
        num_classes (int): Number of output classes for classification. 
                           Required for NNs, ignored by tree-based models.
        average: need it for multiclass
    """

    def __init__(self,  num_classes, average):
        self.num_classes = num_classes
        self.average = average 

    # ---------------- Methods ---------------- #
    def train(self, X_train, y_train, X_val, y_val, config) -> Results:
        """Train a model with the given config and return metrics + model."""
        pass


    # build results
    def _build_results(
        self,
        model,
        train_losses=None,
        train_accs=None,
        val_losses=None,
        val_accs=None,
        hyperparams=None,
    ) -> Results:
        results = Results()

        if train_losses is not None:
            results.train.losses = train_losses
        if train_accs is not None:
            results.train.accs = train_accs
        if val_losses is not None:
            results.val.losses = val_losses
        if val_accs is not None:
            results.val.accs = val_accs
        if hyperparams is not None:
            results.hyperparams = hyperparams

        results.model = model

        return results


