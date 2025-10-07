from .nn_preprocessor import NNPreprocessor
from .tree_preprocessor import TreePreprocessor



class PreprocessorFactory:
    @staticmethod
    def get_preprocessor(model_type, **kwargs):
        model_type = model_type.lower()
        if model_type in ['nn', 'neural', 'neuralnet']:
            return NNPreprocessor(**kwargs)
        elif model_type in ['tree', 'rf', 'randomforest', 'xgboost', 'lgbm']:
            return TreePreprocessor(**kwargs)
        else:
            raise ValueError(f"Unknown model type '{model_type}'")

"""
safest way
class PreprocessorFactory:
    @staticmethod
    def get_preprocessor(model_type: str, cfg: DictConfig):
        model_type = model_type.lower()
        if model_type in ["nn", "neuralnet"]:
            from .nn_preprocessor import NNPreprocessor
            return NNPreprocessor(
                batch_size=cfg.batch_size,
                val_size=cfg.val_size,
                random_state=cfg.random_state
            )
        elif model_type in ["tree", "rf"]:
            from .tree_preprocessor import TreePreprocessor
            return TreePreprocessor(
                do_smote=cfg.do_smote,
                smote_strategy=cfg.smote_strategy
            )
        else:
            raise ValueError(f"Unknown model type {model_type}")
"""