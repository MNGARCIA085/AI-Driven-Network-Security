from .nn_preprocessor import NNPreprocessor
from .tree_preprocessor import TreePreprocessor
from omegaconf import DictConfig





class PreprocessorFactory:
    @staticmethod
    def get_preprocessor(global_cfg, pre_cfg):
        """
        global_cfg: main experiment config (cfg), contains global params like random_seed
        pre_cfg: preprocessing config (cfg.preprocessor)
        """
        model_type = getattr(global_cfg, "model_type", "nn")  # default to nn

        if model_type == "nn":
            return NNPreprocessor(global_cfg, pre_cfg)
        elif model_type in ["tree", "rf"]:
            return TreePreprocessor(global_cfg, pre_cfg)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")





"""
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




