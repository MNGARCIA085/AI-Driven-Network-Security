from .nn_preprocessor import NNPreprocessor
from .tree_preprocessor import TreePreprocessor
from omegaconf import DictConfig




class PreprocessorFactory:
    @staticmethod
    def get_preprocessor(model_type, global_cfg, pre_cfg):
        """
        model_type: nn requires a different prep typa than a tree
        global_cfg: main experiment config (cfg), contains global params like random_seed
        cfg: preprocessing config (cfg.preprocessor)
        """
        
        if model_type == "nn":
            return NNPreprocessor(global_cfg, pre_cfg)
        elif model_type in ["tree", "rf"]:
            return TreePreprocessor(global_cfg, pre_cfg) #rf, xgboost
        else:
            raise ValueError(f"Unknown model_type: {model_type}")




"""
class PreprocessorFactory:
    @staticmethod
    def get_preprocessor(global_cfg, pre_cfg):
        
        global_cfg: main experiment config (cfg), contains global params like random_seed
        pre_cfg: preprocessing config (cfg.preprocessor)
        
        model_type = getattr(global_cfg, "model_type", "nn")  # default to nn

        if model_type == "nn":
            return NNPreprocessor(global_cfg, pre_cfg)
        elif model_type in ["tree", "rf"]:
            return TreePreprocessor(global_cfg, pre_cfg) #rf, xgboost
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

"""






