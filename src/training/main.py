import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.preprocessors.factory import PreprocessorFactory
from src.training.factory import TrainerFactory





mlflow.set_tracking_uri("sqlite:///mlflow.db")






@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # -------------------------------
    # 1️⃣ Print full config
    # -------------------------------
    #print("\n=== Full Config ===")
    #print(OmegaConf.to_yaml(cfg))

    # -------------------------------
    # 2️⃣ Choose model
    # -------------------------------
    # model_name = cfg.models.name
    model_name = cfg.model_type
    print(f"\nSelected model: {model_name}")

    # -------------------------------
    # 3️⃣ Preprocess data
    # -------------------------------
    preprocessor = PreprocessorFactory.get_preprocessor(model_name, cfg, cfg.preprocessor)
    X_train, X_val, y_train, y_val, artifacts = preprocessor.preprocess()
    print(f"Training data shape: {X_train.shape}")

    input_size = preprocessor.input_size
    num_classes = preprocessor.num_classes

    # -------------------------------
    # 4️⃣ Log preprocessing artifacts
    # -------------------------------
    mlflow.set_experiment("IDS_ML_Models")
    with mlflow.start_run(run_name=f"{model_name}_run"):
        """
        for k, v in cfg.models.items():
            mlflow.log_param(k, v)

        for k, v in artifacts.items():
            if k != "X_train":  # skip large arrays
                mlflow.log_param(k, str(v))
        """

        # ---------------------------
        # 5️⃣ Initialize trainer
        # ---------------------------
        trainer = TrainerFactory.get_trainer(model_name, cfg, cfg.models, input_size, num_classes)  
            # returns an instantiated trainer of the corresponding class
        # e.g. TreeTrainer


        # ---------------------------
        # 6️⃣ Train & evaluate
        # ---------------------------
        trainer.train(X_train, y_train, X_val, y_val)
        y_pred = trainer.predict(X_val)

        # ---------------------------
        # 7️⃣ Log metrics
        # ---------------------------
        acc = accuracy_score(y_val, y_pred)
        mlflow.log_metric("val_accuracy", acc)

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, digits=3))


if __name__ == "__main__":
    main()


# python train.py -m models=tree,nn,rf
# python train.py -m models=tree,nn,rf hydra/launcher=submitit_local
# python train.py -m models=tree,nn lr=0.001,0.01 batch_size=32,64


"""
Abstract (trainers handle model-specific logic)

Flexible (preprocessed data passed in)

Tuning-ready (works with Hydra + Ray Tune / Optuna if you add it inside trainer)

MLflow-ready (parameters, artifacts, and metrics logged cleanly)
"""


"""
With model_type (name) I can get:
- preprocessor
- model (it will be used under trainer)
- trainer
- preds
- evals (I think they are common but check)

So it's all about getting factories appropriately
"""


"""
model_name = cfg.models.name
trainer = TrainerFactory.get_trainer(model_name, cfg, cfg.models)

trainer.train(X_train, y_train, X_val, y_val)
y_pred = trainer.evaluate(X_val, y_val)
"""
