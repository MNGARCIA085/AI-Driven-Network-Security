import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import torch

# Adjust these imports to your project
# from your_package.preprocessor import PreprocessorFactory
# from your_package.models import tree_model, nn_model, rf_model

# Map model names to constructors for dynamic selection
model_constructors = {
    "tree": tree_model,
    "nn": nn_model,
    "rf": rf_model
}

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # -----------------------------------------------------------
    # 1️⃣ Print configs for transparency
    # -----------------------------------------------------------
    print("\n=== Full Config ===")
    print(OmegaConf.to_yaml(cfg))

    # -----------------------------------------------------------
    # 2️⃣ Determine the model
    # -----------------------------------------------------------
    model_name = cfg.models.name
    print(f"\nSelected model: {model_name}")

    if model_name not in model_constructors:
        raise ValueError(f"Unknown model: {model_name}")

    ModelClass = model_constructors[model_name]

    # -----------------------------------------------------------
    # 3️⃣ Preprocess data
    # -----------------------------------------------------------
    preprocessor = PreprocessorFactory.get_preprocessor(
        model_name, cfg, cfg.preprocessor
    )
    X_train, X_val, y_train, y_val, artifacts = preprocessor.preprocess()
    print(f"\nTraining data shape: {X_train.shape}")

    # -----------------------------------------------------------
    # 4️⃣ MLflow experiment logging
    # -----------------------------------------------------------
    mlflow.set_experiment("IDS_ML_Models")

    with mlflow.start_run(run_name=f"{model_name}_run"):
        # log model parameters
        for k, v in cfg.models.items():
            mlflow.log_param(k, v)

        # -------------------------------------------------------
        # 5️⃣ Initialize and train model
        # -------------------------------------------------------
        model = ModelClass(cfg, cfg.models)
        model.fit(X_train, y_train)

        # -------------------------------------------------------
        # 6️⃣ Evaluate
        # -------------------------------------------------------
        y_pred = model.predict(X_val)
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, digits=3))

        # log metrics
        from sklearn.metrics import accuracy_score
        mlflow.log_metric("val_accuracy", accuracy_score(y_val, y_pred))

        # optionally save model
        # mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    main()



#python train.py -m models=tree,nn,rf hydra/launcher=submitit_local
#python train.py -m models=tree,nn lr=0.001,0.01 batch_size=32,64

