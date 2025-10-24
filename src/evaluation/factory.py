from .tree import TreeEvaluator
from .nn import NNEvaluator


class EvaluatorFactory:
    @staticmethod
    def get_evaluator(model_type, cfg, model):
        if model_type.lower() == "tree":
            return TreeEvaluator(cfg, model)
        elif model_type.lower() == "nn":
            return NNEvaluator(cfg, model)
        else:
            raise ValueError(f"Unknown tuner type: {model_type}")