# inference/factory.py
import joblib
import mlflow
from .nn import NNPredictor
from .tree import TreePredictor


class PredictorFactory:
    @staticmethod
    def get_predictor(model_type, model, encoder, scaler=None, device="cpu"):
        # Create appropriate predictor
        if model_type == "nn":
            return NNPredictor(model, encoder, scaler, device=device)
        elif model_type=='tree':
            return TreePredictor(model, encoder)
        else:
            raise ValueError(f"Unknown tuner type: {model_type}")





