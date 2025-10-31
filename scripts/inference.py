
import hydra
import mlflow
import joblib
import pandas as pd
from omegaconf import DictConfig
from src.utils.model_selection import select_best_model
from src.preprocessors.factory import PreprocessorFactory 
from src.inference.factory import PredictorFactory


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # load inference data
    path = cfg.preprocessor.path_inference
    df = pd.read_csv(path)

    # Load data from the best run
    #results = select_best_model("nn_experiment", model_type = "nn")
    results = select_best_model("nn_experiment", model_type='nn')
    model_type = results["model_type"]

    # Common load
    encoder = joblib.load(results["encoder_path"])

    # Depending on the type
    if model_type == "nn":
        model = mlflow.pytorch.load_model(results["model_uri"])
        scaler = joblib.load(results["scaler_path"])

        data_prep = {
            "df": df,
            "scaler": scaler,
        }

        data_pred = {
            "model_type": model_type,
            "model": model,
            "scaler": scaler,
            "encoder": encoder,
            "device": "cpu"
        }

    else:  # tree models
        model = mlflow.sklearn.load_model(results["model_uri"])

        data_prep = {
            "df": df,
        }

        data_pred = {
            "model_type": model_type,
            "model": model,
            "encoder": encoder,
        }


        
    # Preprocess data
    preprocessor = PreprocessorFactory.get_preprocessor(model_type, cfg, cfg.preprocessor) 
    X_values = preprocessor.preprocess_inference(**data_prep)

    # get predictor
    predictor = PredictorFactory.get_predictor(**data_pred)


    # preds
    preds = predictor.predict(X_values)
    print(preds)
    print(preds.shape) # number of samples x 1 (predicted class, BENINGN, DoS.....)

    # preds as probs
    probs = predictor.predict_proba(X_values)
    print(probs)
    print(probs.shape) # number of samples x number of labels


    


if __name__ == "__main__":
    main()





# prep. for a single example
# preprocess single
#import numpy as np
#sample = np.random.rand(78)
#preprocessor = PreprocessorFactory.get_preprocessor(model_type, cfg, cfg.preprocessor)
#X_sample = preprocessor.preprocess_single(sample, scaler)


"""
pred:

logits -> tensor([[  159.9129,  -141.3121,  -654.4407,   -77.6352,  -407.2115,  -491.3810,
          -748.4187,  -302.7671,    97.4854, -1398.7358]])


probs -> tensor([[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 1.0840e-22, 0.0000e+00]])

class predicted -> tensor([0])


"""