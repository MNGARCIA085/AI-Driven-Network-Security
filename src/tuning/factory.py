class TunerFactory:
    @staticmethod
    def create(tuner_type, cfg, X_train, y_train, X_val, y_val, **kwargs): # tuner_type should be model_type
        if tuner_type.lower() == "tree":
            from tuners.tree_tuner import TreeTuner
            return TreeTuner(cfg, X_train, y_train, X_val, y_val)
        elif tuner_type.lower() == "nn":
            from tuners.nn_tuner import NNTuner
            num_classes = kwargs.get("num_classes")
            if num_classes is None:
                raise ValueError("num_classes must be provided for NNTuner")
            return NNTuner(cfg, X_train, y_train, X_val, y_val, num_classes)
        else:
            raise ValueError(f"Unknown tuner type: {tuner_type}")


"""
tuner = TunerFactory.create(
    tuner_type="nn",
    cfg=cfg_nn,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    num_classes=3
)

best_cfg = tuner.tune(num_samples=5)
metrics = tuner.train_best_model(best_cfg)
"""