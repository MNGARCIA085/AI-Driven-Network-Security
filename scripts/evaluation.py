import pandas as pd 
import joblib
import hydra
import mlflow
import os
import numpy as np
from omegaconf import DictConfig, OmegaConf
from src.preprocessors.factory import PreprocessorFactory 
from src.inference.factory import PredictorFactory
from src.utils.logging import log_test_results, select_best_model
from src.evaluation.base import Evaluator
from src.utils.results import TestResults, Metrics
from src.config.data import build_data_config


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Load and preprocess data
    path = cfg.preprocessor.path_test
    df = pd.read_csv(path)

    # 2. Load model, scaler, encoder from the best run
    results = select_best_model(cfg.experiment_name) #, model_type="nn")
    tuning_run_id = results['run_id']
    model_type = results['model_type'] 

    # common load
    encoder = joblib.load(results['encoder_path'])
    
    # depending on the type
    if model_type == 'nn':
        model = mlflow.pytorch.load_model(results['model_uri'])
        scaler = joblib.load(results['scaler_path'])

        # data for prep.
        data_prep = {
            "df": df,
            "scaler": scaler,
            "label_encoder": encoder,
        }


        # data for pred
        data_pred = {
            "model_type": model_type,
            "model": model,
            "encoder": encoder,
            "device": "cpu"
        }

    else: # trees
        model = mlflow.sklearn.load_model(results['model_uri'])
        
        data_prep = {
            "df": df,
            "label_encoder": encoder,
        }

        # data for pred
        data_pred = {
            "model_type": model_type,
            "model": model,
            "encoder": encoder,
        }     


    # 3. Preprocess test data
    data_cfg = build_data_config(cfg.preprocessor, cfg)
    preprocessor = PreprocessorFactory.get_preprocessor(model_type,data_cfg)
    X_values, y_encoded = preprocessor.preprocess_test(**data_prep)


    # 4. Make preds with the best model
    predictor = PredictorFactory.get_predictor(**data_pred)
    y_pred = predictor.predict(X_values)
    y_prob = predictor.predict_proba(X_values)

    #import numpy as np
    #print(np.unique(y_encoded)); y_encoded is like 0 1 7 2..........
    

    # 5. Run evaluator
    evaluator = Evaluator()
    metrics = evaluator.compute_metrics(y_encoded, y_pred)



    # 6. Format results
    results = TestResults(
            metrics = Metrics(**metrics),
            preds = np.array(y_pred),
            probs = np.array(y_prob),
            labels = np.array(y_encoded)
        )

    # Logging
    log_test_results(cfg.experiment_name, tuning_run_id, model_type, results)




# Execution block
if __name__ == "__main__":
    main()