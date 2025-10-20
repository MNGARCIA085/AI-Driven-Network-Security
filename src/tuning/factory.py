from .tree import TreeTuner
from .nn import NNTuner



class TunerFactory:
    @staticmethod
    def get_tuner(model_type, cfg, X_train, y_train, X_val, y_val, **kwargs):
        if model_type.lower() == "tree":
            return TreeTuner(cfg, X_train, y_train, X_val, y_val)
        elif model_type.lower() == "nn":
            num_classes = kwargs.get("num_classes")
            if num_classes is None:
                raise ValueError("num_classes must be provided for NNTuner")
            return NNTuner(cfg, X_train, y_train, X_val, y_val, num_classes)
        else:
            raise ValueError(f"Unknown tuner type: {tuner_type}")


