from src.utils.model_selection import select_best_model
#from inference.factory import PredictorFactory
import hydra
from omegaconf import DictConfig
import mlflow
import joblib


import pandas as pd

from src.preprocessors.factory import PreprocessorFactory 


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):


    path = cfg.preprocessor.path_inference
    df = pd.read_csv(path)


    # Load model, scaler, encoder from the best run
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

    # Select and apply prep
    


    

    preprocessor = PreprocessorFactory.get_preprocessor(model_type, cfg, cfg.preprocessor) 
    X_values = preprocessor.preprocess_inference(**data_prep)



    # to test preprocessor
    from .factory import PredictorFactory

    predictor = PredictorFactory.get_predictor(**data_pred)



    preds = predictor.predict(X_values)
    print(preds)

    probs = predictor.predict_proba(X_values)
    print(probs)

    print(probs.shape) # number of samples x number of labels


    ###############################
    ### TREE TEST #################
    ###############################
    # TODO: add tree predictor example here


    """ok
    from .tree import TreePredictor
    predictor = TreePredictor(model, encoder)
    pred_one = predictor.predict(X_values)  
    print(pred_one)  
    """


    ##############################
    ### NN TEST ##################
    ##############################


    """ OK
    from .nn import NNPredictor
    predictor = NNPredictor(model, encoder, scaler, device='cpu')
    pred_one = predictor.predict(X_values)
    print(pred_one)
    """
    #pred_batch = predictor.predict(X_batch)
    #probs = predictor.predict_proba(X_batch)

    


if __name__ == "__main__":
    main()





# prep. for a single example
# preprocess single
#import numpy as np
#sample = np.random.rand(78)
#preprocessor = PreprocessorFactory.get_preprocessor(model_type, cfg, cfg.preprocessor)
#X_sample = preprocessor.preprocess_single(sample, scaler)
