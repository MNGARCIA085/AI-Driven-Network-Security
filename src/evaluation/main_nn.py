import pandas as pd 
import mlflow.pytorch 
import joblib

# Assuming these are correct relative imports for your project structure
from .nn import NNEvaluator
from .model_selection import select_best_model
from src.preprocessors.factory import PreprocessorFactory 




import hydra
from omegaconf import DictConfig, OmegaConf



mlflow.set_tracking_uri("sqlite:///mlflow.db")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Load and preprocess data
    path = "data/test_data.csv"
    df = pd.read_csv(path)

    # 2. Load model, scaler, encoder from the best run
    model_uri, scaler_path, encoder_path = select_best_model("nn_experiment")
    model = mlflow.pytorch.load_model(model_uri)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    print(model)
    #print(scaler.mean_)


    # 3. Preprocess test data
    model_type = 'nn'
    preprocessor = PreprocessorFactory.get_preprocessor(model_type, cfg, cfg.preprocessor) 
    
    # X_values is the feature matrix, y_encoded is the target variable
    X_values, y_encoded = preprocessor.preprocess_test(df, scaler, encoder)



    #print(X_values)
    #print(y_encoded)


    # 4. Run nn evaluator
    evaluator = NNEvaluator(model)
    
    # Original used X_test, y_test; corrected to X_values, y_encoded
    f1, loss, acc, preds, labels, probs = evaluator.evaluate(X_values, y_encoded)

    print(f"Evaluation Results:\nF1-Score: {f1}\nLoss: {loss}\nAccuracy: {acc}")
    # Optional: Save predictions or other artifacts here



# Corrected the execution block
if __name__ == "__main__":
    main()