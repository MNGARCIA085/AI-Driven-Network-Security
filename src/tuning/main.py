from .nn import NNTuner
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf


from src.preprocessors.factory import PreprocessorFactory

import matplotlib.pyplot as plt
import os
import joblib




mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set default folder for artifacts
artifact_dir = os.path.abspath("./mlruns")  # choose any folder
os.makedirs(artifact_dir, exist_ok=True)


mlflow.set_experiment("nn_experiment")  # ensures artifact path is set



@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # add logging

    # get data from prep
    # model_name = cfg.models.name
    model_name = cfg.model_type
    print(f"\nSelected model: {model_name}")


    with mlflow.start_run(run_name='vamos'):

        # -------------------------------
        # 3️⃣ Preprocess data
        # -------------------------------
        preprocessor = PreprocessorFactory.get_preprocessor(model_name, cfg, cfg.preprocessor)
        X_train, X_val, y_train, y_val, artifacts = preprocessor.preprocess()
        print(f"Training data shape: {X_train.shape}")

        input_size = preprocessor.input_size
        num_classes = preprocessor.num_classes



        artifacts = preprocessor.get_artifacts()
        # Log numeric and string metadata
        mlflow.log_param("balance_factor", artifacts["balance_factor"])
        mlflow.log_param("val_size", artifacts["val_size"])
        # Log class distributions as JSON
        mlflow.log_dict(artifacts["class_dist_before_smote"], "class_dist_before_smote.json")
        mlflow.log_dict(artifacts["class_dist_after_smote"], "class_dist_after_smote.json")
        # Optionally log shapes
        mlflow.log_param("train_shape", str(artifacts["train_shape"]))
        mlflow.log_param("val_shape", str(artifacts["val_shape"]))
        scaler = artifacts["scaler"]
        filename = "scaler.pkl"
        # Save + log + clean up
        joblib.dump(scaler, filename)
        mlflow.log_artifact(filename, artifact_path="preprocessor")
        os.remove(filename)




        # 1️⃣ Tune
        tuner = NNTuner(X_train, y_train, X_val, y_val, num_classes)
        best_config = tuner.tune(num_samples=10)

        # 2️⃣ Train best model and get all metrics
        metrics = tuner.train_best_model(best_config, epochs=5)
        #print(metrics)

        print(metrics)
        print(type(metrics))

        # 3️⃣ External logging (MLflow, wandb, etc.)
        # e.g., mlflow.log_metrics(metrics) and generate plots from metrics["train_losses"], etc.

        

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])

        #
        #train_val_loss_path = os.path.join(os.getcwd(), "loss_curve.png")
        #val_acc_path = os.path.join(os.getcwd(), "val_acc_curve.png")




        # --- Log training/validation curves ---
        # Training loss curve
        plt.figure()
        plt.plot(metrics["train_losses"], label="train_loss")
        plt.plot(metrics["val_losses"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        train_val_loss_path = "loss_curve.png"
        plt.savefig(train_val_loss_path)
        plt.close()
        mlflow.log_artifact(train_val_loss_path)

        # Validation accuracy curve
        plt.figure()
        plt.plot(metrics["val_accs"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        val_acc_path = "val_acc_curve.png"
        plt.savefig(val_acc_path)
        plt.close()
        mlflow.log_artifact(val_acc_path)

        # --- Log the model ---
        #
        # after metrics computed
        mlflow.pytorch.log_model(metrics["model"], artifact_path="model") # it is saved in models


        # Optionally, remove local files after logging
        os.remove(train_val_loss_path)
        os.remove(val_acc_path)


if __name__ == "__main__":
    main()
