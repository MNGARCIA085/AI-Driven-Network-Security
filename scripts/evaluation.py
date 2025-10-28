import pandas as pd 
import joblib
import hydra
import mlflow.pytorch 
import mlflow.sklearn
import os
from omegaconf import DictConfig, OmegaConf
from src.preprocessors.factory import PreprocessorFactory 
from src.evaluation.model_selection import select_best_model
from src.evaluation.factory import EvaluatorFactory
from src.utils.logging import log_test_results


mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set default folder for artifacts
artifact_dir = os.path.abspath("./mlruns")  # choose any folder
os.makedirs(artifact_dir, exist_ok=True)

# ensures artifact path is set
mlflow.set_experiment("nn_experiment")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Load and preprocess data
    path = cfg.preprocessor.path_test
    df = pd.read_csv(path)

    # 2. Load model, scaler, encoder from the best run
    results = select_best_model("nn_experiment")
    model_type = results['model_type'] 

    # common load
    encoder = joblib.load(results['encoder_path'])
    
    # depending on the type
    if model_type == 'nn':
        model = mlflow.pytorch.load_model(results['model_uri'])
        scaler = joblib.load(results['scaler_path'])

        data_prep = {
            "df": df,
            "scaler": scaler,
            "label_encoder": encoder,
        }
    else: # trees
        model = mlflow.sklearn.load_model(results['model_uri'])
        
        data_prep = {
            "df": df,
            "label_encoder": encoder,
        }        


    # 3. Preprocess test data
    preprocessor = PreprocessorFactory.get_preprocessor(model_type, cfg, cfg.preprocessor) 
    X_values, y_encoded = preprocessor.preprocess_test(**data_prep)


    # 4. Run evaluator
    evaluator = EvaluatorFactory.get_evaluator(model_type, cfg.evaluation, model)
    
    # Original used X_test, y_test; corrected to X_values, y_encoded
    results = evaluator.evaluate(X_values, y_encoded)


    # Logging
    log_test_results(model_type, results)











# Corrected the execution block
if __name__ == "__main__":
    main()