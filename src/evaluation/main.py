import pandas as pd 
import mlflow.pytorch 
import joblib

# Assuming these are correct relative imports for your project structure
from .nn import NNEvaluator
from .model_selection import select_best_model
from src.preprocessors.factory import PreprocessorFactory 




import hydra
from omegaconf import DictConfig, OmegaConf


from .factory import EvaluatorFactory


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
    

    if model_type == 'nn':
        model = mlflow.pytorch.load_model(results['model_uri'])
    else:
        model = mlflow.sklearn.load_model(results['model_uri'])
    

    if results['scaler_path'] is not None:
        scaler = joblib.load(results['scaler_path']) # it may be None
    


    encoder = joblib.load(results['encoder_path'])

    print(model_type)




    data = {
        "df":df,
        "label_encoder":encoder
    }


    # 3. Preprocess test data
    preprocessor = PreprocessorFactory.get_preprocessor(model_type, cfg, cfg.preprocessor) 
    # ort if isinsance
    if model_type == 'nn':
        X_values, y_encoded = preprocessor.preprocess_test(df, scaler, encoder)
    else:
        #X_values, y_encoded = preprocessor.preprocess_test(df, encoder)
        X_values, y_encoded = preprocessor.preprocess_test(**data)


    # Run evaluator


    # 4. Run nn evaluator
    evaluator = EvaluatorFactory.get_evaluator(model_type, model)
    
    # Original used X_test, y_test; corrected to X_values, y_encoded
    results = evaluator.evaluate(X_values, y_encoded)
    print(results)



    print(model_type)

    # roc curve and cm if needed




# main per run not really needed that can go in selection





# Corrected the execution block
if __name__ == "__main__":
    main()