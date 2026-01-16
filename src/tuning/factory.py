from .tree import TreeTuner
from .nn import NNTuner




class TunerFactory:
    @staticmethod
    def get_tuner(model_type, cfg, X_train, y_train, X_val, y_val, preprocessor):
        if model_type.lower() == "tree":
            return TreeTuner(cfg, X_train, y_train, X_val, y_val, preprocessor.num_classes)
        elif model_type.lower() == "nn":
            return NNTuner(cfg, X_train, y_train, X_val, y_val, preprocessor.num_classes)
        else:
            raise ValueError(f"Unknown tuner type: {model_type}")


