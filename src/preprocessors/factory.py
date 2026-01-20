from .nn_preprocessor import NNPreprocessor
from .tree_preprocessor import TreePreprocessor
from omegaconf import DictConfig




class PreprocessorFactory:
    @staticmethod
    def get_preprocessor(model_type, data_cfg):
        """
        model_type: nn requires a different prep. type than a tree
        """
        
        if model_type == "nn":
            return NNPreprocessor(data_cfg)
        elif model_type in ["tree", "rf"]:
            return TreePreprocessor(data_cfg) #rf, xgboost
        else:
            raise ValueError(f"Unknown model_type: {model_type}")







