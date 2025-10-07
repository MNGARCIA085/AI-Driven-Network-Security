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
