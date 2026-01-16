import ray
from abc import ABC, abstractmethod
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from src.utils.results import Results, Metrics



class BaseTrainer():
    """
    Base training class
    """

    def __init__(self,  average):
        self.average = average  # common for metric computations; but it shuldnnt be here probably

    # ---------------- Methods ---------------- #
    def train(self, config) -> Results:
        """Train a model with the given config and return metrics + model."""
        pass


    # build results; take metrics later!!!!!!!!!!
    def _build_results(
        self,
        model,
        train_losses=None,
        train_accs=None,
        val_losses=None,
        val_accs=None,
        val_preds=None,
        val_labels=None,
        val_probs=None,
        val_metrics=None,
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
        if val_preds is not None:
            results.val.preds = val_preds
        if val_labels is not None:
            results.val.labels = val_labels
        if val_probs is not None:
            results.val.probs = val_probs
        if val_metrics is not None:
            results.val.metrics = Metrics(**val_metrics)

        if hyperparams is not None:
            results.hyperparams = hyperparams

        results.model = model
        return results


