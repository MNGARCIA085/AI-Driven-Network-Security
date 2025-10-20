import ray
from abc import ABC, abstractmethod
from ray import tune
from ray.tune.schedulers import ASHAScheduler

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
    def train_best_model(self, config):
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


