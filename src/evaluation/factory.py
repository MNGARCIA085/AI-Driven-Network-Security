from .tree import TreeEvaluator
from .nn import NNEvaluator


class EvaluatorFactory:
    @staticmethod
    def get_evaluator(model_type, model):
        if model_type.lower() == "tree":
            return TreeEvaluator(model)
        elif model_type.lower() == "nn":
            return NNEvaluator(mddel)
        else:
            raise ValueError(f"Unknown tuner type: {model_type}")