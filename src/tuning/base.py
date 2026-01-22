import ray
from abc import ABC, abstractmethod
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from src.utils.results import Results, Metrics



class BaseTuner(): # ABC
    """
    Abstract base class for Ray Tune-based hyperparameter tuning.
    Handles Ray object storage, generic tune(), and metric averaging.
    """

    def __init__(self, cfg, X_train, y_train, X_val, y_val, num_classes):
        self.cfg = cfg
        self.X_train_id = ray.put(X_train)
        self.y_train_id = ray.put(y_train)
        self.X_val_id = ray.put(X_val)
        self.y_val_id = ray.put(y_val)
        self.num_classes = num_classes


    # ---------------- Methods ---------------- #
    @staticmethod
    def train_model_ray(config, X_train_id, y_train_id, X_val_id, y_val_id, average, num_classes): # full signature or use config 
        """Train one model configuration for Ray Tune."""
        pass

    @abstractmethod
    def get_tune_config(self):
        """Return the hyperparameter search space for Ray Tune."""
        pass


    # ---------------- Shared method ---------------- #
    def tune(self): # self is ok here
        """
        Generic Ray Tune wrapper.
        Subclasses provide _train_model_ray and get_tune_config.


        Note. Arguments passed through tune.with_parameters are already deserialized
        when received by the train function, even if they originate from ray.put().
        Avoid calling ray.get() again inside train_model_ray.


        """
        config = self.get_tune_config()
        scheduler = ASHAScheduler(metric="f1", mode="max")

       
        # train_fn
        train_fn = type(self).train_model_ray 


        tuner = tune.Tuner(
            tune.with_parameters(
                train_fn,
                X_train=self.X_train_id,
                y_train=self.y_train_id,
                X_val=self.X_val_id,
                y_val=self.y_val_id,
                num_classes=self.num_classes,
                average=self.cfg.average,
            ),
            param_space=config,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=self.cfg.num_samples,
            ),
        )


        results = tuner.fit()
        best = results.get_best_result(metric="f1", mode="max")
        return best.config



    


