import ray
from abc import ABC, abstractmethod
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from src.utils.results import Results, Metrics



class BaseTuner(ABC):
    """
    Abstract base class for Ray Tune-based hyperparameter tuning.
    Handles Ray object storage, generic tune(), and metric averaging.
    """

    def __init__(self, cfg, X_train, y_train, X_val, y_val):
        self.cfg = cfg
        self.X_train_id = ray.put(X_train)
        self.y_train_id = ray.put(y_train)
        self.X_val_id = ray.put(X_val)
        self.y_val_id = ray.put(y_val)
        self.average = cfg.average  # common for metric computations

    # ---------------- Abstract methods ---------------- #
    @abstractmethod
    def _train_model_ray(self, config):
        """Train one model configuration for Ray Tune."""
        pass

    @abstractmethod
    def get_tune_config(self):
        """Return the hyperparameter search space for Ray Tune."""
        pass

    @abstractmethod
    def train_best_model(self, config) -> Results:
        """Train a model with the given config and return metrics + model."""
        pass

    #@abstractmethod
    #def eval_model(self, model, X, y):
    #    """Evaluate a trained model."""
    #    pass

    # ---------------- Shared method ---------------- #
    def tune(self, num_samples=5): # -> ok
        """
        Generic Ray Tune wrapper.
        Subclasses provide _train_model_ray and get_tune_config.
        """
        config = self.get_tune_config()
        scheduler = ASHAScheduler(metric="f1", mode="max")

        tuner = tune.Tuner(
            tune.with_parameters(self._train_model_ray),
            param_space=config,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=num_samples
            )
        )
        results = tuner.fit()
        best = results.get_best_result(metric="f1", mode="max")
        return best.config



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


