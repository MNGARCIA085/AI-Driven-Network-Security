"""
stage        = train | tune | eval
model_type   = nn | tree | xgboost | ...
dataset      = dataset_name_or_version
task         = binary_classification | multiclass

"""








import mlflow
import mlflow.sklearn
import joblib
import os
import json
from mlflow.tracking import MlflowClient
from pathlib import Path
from .plots import plot_cm, plot_roc, plot_train_val
from dataclasses import is_dataclass, asdict


# Project root (2 levels up from this file)
root_dir = Path(__file__).resolve().parents[2]

# Tracking DB
mlflow.set_tracking_uri(f"sqlite:///{root_dir / 'mlflow.db'}")

# Artifacts folder
artifact_dir = root_dir / "mlruns"
os.makedirs(artifact_dir, exist_ok=True)




# tags
def log_tags(stage, model_type, dataset, extra_tags=None):

    tags = {
        "stage": stage,
        "model_type": model_type,
        "task": "multiclass_classification",
        "dataset": dataset,
        "data_version": "v1",

    }

    if extra_tags:
        tags.update(extra_tags)
    mlflow.set_tags(tags)


# comon params
def log_common_params(artifacts):
    mlflow.log_param("balance_factor", artifacts["balance_factor"])
    mlflow.log_param("val_size", artifacts["val_size"])
    mlflow.log_param("train_shape", artifacts["train_shape"])
    mlflow.log_param("val_shape", artifacts["val_shape"])



# features, encoder and scaler
def log_preprocessors(artifacts):
    mlflow.log_dict({"features": artifacts["features"]}, "features.json")
                # -> remove later appropiately to avoid race conditions!!!!!
    mlflow.log_dict(artifacts["class_dist_before_smote"], "class_dist_before_smote.json")
    mlflow.log_dict(artifacts["class_dist_after_smote"], "class_dist_after_smote.json")


    # encoder
    encoder = artifacts["encoder"]
    filename = "label_encoder.pkl"
    joblib.dump(encoder, filename)
    mlflow.log_artifact(filename, artifact_path="preprocessor")
    os.remove(filename)
    mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    mlflow.log_dict(mapping, "encoder_mapping.json")


    # scaler (only for NNs)
    if artifacts.get("scaler") is not None:
        filename = "scaler.pkl"
        joblib.dump(artifacts["scaler"], filename)
        mlflow.log_artifact(filename, artifact_path="preprocessor")
        os.remove(filename)

        # scaler type
        mlflow.log_param("scaler_type", artifacts["scaler_type"])



# metrics
def log_metrics(metrics):
    """
    metrics: dict or dataclass / object with attributes
    """
    # Normalize to dict
    if isinstance(metrics, dict):
        metrics_dict = metrics
    elif is_dataclass(metrics):
        metrics_dict = asdict(metrics)
    else:
        # generic object with attributes
        metrics_dict = vars(metrics)

    # Log only valid numeric metrics
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v)



# hyperparams
def log_hyperparams(results):
    mlflow.log_params(results.hyperparams)


# models
def log_model(results, model_type):
    if model_type == "nn":
        mlflow.pytorch.log_model(results.model, artifact_path="model")
    else:
        mlflow.sklearn.log_model(results.model, artifact_path="model")


# CM and ROC curves
def log_plots(results, model_type, train_curves=False):
    cm = plot_cm(results.labels, results.preds)
    mlflow.log_artifact(cm)
    os.remove(cm)

    roc = plot_roc(results.labels, results.probs)
    mlflow.log_artifact(roc)
    os.remove(roc)


# training curves
def log_training_curves(train_data, val_data, filename, title):
    loss_path = plot_train_val(train_data, val_data, filename, title)
    mlflow.log_artifact(loss_path)
    os.remove(loss_path)



# later -> add tuning trials!!!


# log exp. log_Experiment
def logging(exp_name, run_name, artifacts, results, model_type, stage):
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=run_name):
        log_tags(stage, model_type, "train/val")
        log_common_params(artifacts)
        log_preprocessors(artifacts)
        log_metrics(results.val.metrics)
        log_hyperparams(results)
        log_model(results, model_type)
        log_plots(results.val, model_type, train_curves=True)
        if model_type == "nn":
            log_training_curves(results.train.losses, results.val.losses, "loss_curve.png", 'Loss')
            log_training_curves(results.train.accs, results.val.accs, "acc_curve.png", 'Accuracy')








#------------------Final evaluation (with test set)-----------------------------
# log test results (eval!!!!!)
def log_test_results(exp_name, tuning_run_id, model_type, results, stage='eval'):
    # ensures artifact path is set
    mlflow.set_experiment(exp_name)
    
    # run
    with mlflow.start_run(run_name="test_evaluation"):
        log_tags(stage, model_type, "test", {"tuning_run_id": tuning_run_id})
        #mlflow.set_tag("tuning_run_id", tuning_run_id)
        log_metrics(results.metrics)
        log_plots(results, model_type)











#----------------to adapt later--------------------------------------
# get best model data
def select_best_model(experiment_name, metric="f1", model_type=None, data_version="v1"):
    """
    Select the best model overall or the best one of a specific type.
    
    Args:
        experiment_name (str): Name of the MLflow experiment.
        metric (str): Metric used for ranking (default: 'f1').
        model_type (str, optional): Filter by model type (e.g., 'nn' or 'tree').
        data_version (str): Version tag of the data (default: 'v1').

    Returns:
        dict: Paths and metadata of the best run.
    """

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # Build filter
    filter_str = f"run_name = 'Tuning' AND tags.data_version = '{data_version}'"
    if model_type:
        filter_str += f" AND tags.model_type = '{model_type}'"

    # Search runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_str,
        order_by=[f"metrics.{metric} DESC"],
    )

    if not runs:
        raise ValueError(f"No runs found for model_type='{model_type}' and data_version='{data_version}'.")

    # Take best run
    best_run = runs[0]
    run_id = best_run.info.run_id

    print(f"Best {model_type or 'overall'} run ID: {run_id}")
    print(f"Best {metric.upper()}: {best_run.data.metrics[metric]:.4f}")

    # Common part
    model_uri = f"runs:/{run_id}/model"
    encoder_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="preprocessor/label_encoder.pkl"
    )


    model_type = best_run.data.tags.get("model_type")

    # NN models include scaler
    if model_type == "nn":
        scaler_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="preprocessor/scaler.pkl"
        )
        return {
            "run_id": run_id,
            "model_type": model_type,
            "model_uri": model_uri,
            "scaler_path": scaler_path,
            "encoder_path": encoder_path,
        }

    # Tree models
    return {
        "run_id": run_id,
        "model_type": model_type,
        "model_uri": model_uri,
        "encoder_path": encoder_path,
    }






