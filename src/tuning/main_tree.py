from .tree import TreeTuner
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from src.preprocessors.factory import PreprocessorFactory
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

import mlflow.sklearn



mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set default folder for artifacts
artifact_dir = os.path.abspath("./mlruns")  # choose any folder
os.makedirs(artifact_dir, exist_ok=True)


mlflow.set_experiment("nn_experiment")  # ensures artifact path is set














#model = metrics['model']
#mlflow.sklearn.log_model(model, "tree_model")
# to fix warning:mlflow.sklearn.log_model(model, name="tree_model")
#model = mlflow.sklearn.load_model("runs:/<run_id>/tree_model")




from .main import plot_cm,plot_roc,plot_acc,plot_loss



# later test !!!!!!!!!!!!!!!!!!!!!!!!
def logging(artifacts, results, model_type="nn"):
    # Common params
    mlflow.log_param("val_size", artifacts["val_size"])

    # Optional artifacts
    if "scaler" in artifacts and artifacts["scaler"] is not None:
        filename = "scaler.pkl"
        joblib.dump(artifacts["scaler"], filename)
        mlflow.log_artifact(filename, artifact_path="preprocessor")
        os.remove(filename)

    # Metrics (shared)
    for m in ["accuracy", "precision", "recall", "f1"]:
        if m in results:
            mlflow.log_metric(m, results[m])

    # Model-specific logs
    if model_type == "nn":
        mlflow.pytorch.log_model(results["model"], artifact_path="model")
        for plot_fn in [plot_loss, plot_acc, plot_cm, plot_roc]:
            path = plot_fn(results)
            mlflow.log_artifact(path)
            os.remove(path)
    else:
        mlflow.sklearn.log_model(results["model"], name="model")
        for plot_fn in [plot_cm, plot_roc]:
            path = plot_fn(results)
            mlflow.log_artifact(path)
            os.remove(path)















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