import pandas as pd 
import mlflow.pytorch 
import joblib

# Assuming these are correct relative imports for your project structure
from .tree import TreeEvaluator
from .model_selection import select_best_model
from src.preprocessors.factory import PreprocessorFactory 

import hydra
from omegaconf import DictConfig, OmegaConf



import mlflow.sklearn


mlflow.set_tracking_uri("sqlite:///mlflow.db")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Load and preprocess data
    path = "data/test_data.csv"
    df = pd.read_csv(path)

    # 2. Load model, scaler, encoder from the best run
    results = select_best_model("nn_experiment")
    
    model_type = results['model_type']
    model = mlflow.sklearn.load_model(results['model_uri'])
    print(model)
    #scaler = joblib.load(results['scaler_path'])
    encoder = joblib.load(results['encoder_path'])

    print(model_type)
    model_type = 'tree'

    # 3. Preprocess test data
    preprocessor = PreprocessorFactory.get_preprocessor(model_type, cfg, cfg.preprocessor) 
    
    # X_values is the feature matrix, y_encoded is the target variable
    X_values, y_encoded = preprocessor.preprocess_test(df, encoder) # before -> scaler, encoder


    # 4. Run evaluator
    evaluator = TreeEvaluator(model)
    
    # Original used X_test, y_test; corrected to X_values, y_encoded
    results = evaluator.evaluate(X_values, y_encoded)
    print(results)

    # roc curve and cm if needed








# Corrected the execution block
if __name__ == "__main__":
    main()