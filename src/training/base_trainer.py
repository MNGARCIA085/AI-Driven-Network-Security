



class BaseTrainer:
    """
    Abstract base trainer class.
    Trainer receives preprocessed data in train/evaluate methods.
    """
    def __init__(self, cfg, model_cfg, input_size, num_classes):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.input_size = input_size
        self.num_classes = num_classes
        #self.model = self._create_model()

    def _create_model(self):
        raise NotImplementedError("Subclasses must implement _create_model")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model.
        X_val, y_val optional for models that support validation (like NNs).
        """
        raise NotImplementedError("Subclasses must implement train")

    def evaluate(self, X_val, y_val):
        """
        Evaluate the model and return predictions.
        """
        raise NotImplementedError("Subclasses must implement evaluate")



"""
✅ Advantages of this design

Trainer only deals with training & evaluation.

Preprocessing and artifacts logging are fully outside.

Tree and NN logic is separate — NNs can have epochs, validation, early stopping, etc.

Easy to extend for new model types: just add a new Trainer subclass and update the factory.

Works seamlessly with Hydra multiruns + MLflow + hyperparameter tuning.
"""


