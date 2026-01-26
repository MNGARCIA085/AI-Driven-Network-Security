import mlflow
import os
from pathlib import Path



def init_mlflow(exp_name):
    # Project root (2 levels up from this file)
    root_dir = Path(__file__).resolve().parents[3]

    # Tracking DB
    mlflow.set_tracking_uri(f"sqlite:///{root_dir / 'mlflow.db'}")

    # Artifacts folder
    artifact_dir = root_dir / "mlruns"
    os.makedirs(artifact_dir, exist_ok=True)

    # set experimnet name
    mlflow.set_experiment(exp_name)





"""
stage        = train | tune | eval
model_type   = nn | tree | xgboost | ...
dataset      = dataset_name_or_version
task         = binary_classification | multiclass


mlflow.set_tag("stage", "train")   # during normal training
mlflow.set_tag("stage", "tune")    # Ray Tune runs
mlflow.set_tag("stage", "eval")    # test set only


"""