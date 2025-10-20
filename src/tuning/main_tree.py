from .tree import TreeTuner
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from src.preprocessors.factory import PreprocessorFactory
import os
from src.utils.logging import logging


mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set default folder for artifacts
artifact_dir = os.path.abspath("./mlruns")  # choose any folder
os.makedirs(artifact_dir, exist_ok=True)


mlflow.set_experiment("nn_experiment")  # ensures artifact path is set



#model = metrics['model']
#mlflow.sklearn.log_model(model, "tree_model")
# to fix warning:mlflow.sklearn.log_model(model, name="tree_model")
#model = mlflow.sklearn.load_model("runs:/<run_id>/tree_model")




@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # add logging

    # get data from prep
    # model_name = cfg.models.name
    model_name = cfg.model_type
    print(f"\nSelected model: {model_name}")


    with mlflow.start_run(run_name='vamosTree'):

        # -------------------------------
        # Preprocess data
        # -------------------------------
        preprocessor = PreprocessorFactory.get_preprocessor(model_name, cfg, cfg.preprocessor)
        X_train, X_val, y_train, y_val, artifacts = preprocessor.preprocess()
        print(f"Training data shape: {X_train.shape}")
        artifacts = preprocessor.get_artifacts()


        # Tune
        model_type = model_name
        cfg_tuning = OmegaConf.load(f"config/tuning/{model_type}.yaml") #use tuning/tree.yaml
        tuner = TreeTuner(cfg_tuning, X_train, y_train, X_val, y_val)
        



        best_config = tuner.tune(num_samples=5)

        # Train best model and get all metrics
        results = tuner.train_best_model(best_config)

        # 3. External logging (MLflow, wandb, etc.)
        logging(artifacts, results, 'Tree')
        


if __name__ == "__main__":
    main()



#python -m src.tuning.main_tree model_type=tree