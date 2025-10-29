import joblib
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")




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
            "model_type": model_type,
            "model_uri": model_uri,
            "scaler_path": scaler_path,
            "encoder_path": encoder_path,
        }

    # Tree models
    return {
        "model_type": model_type,
        "model_uri": model_uri,
        "encoder_path": encoder_path,
    }

















# nn_experiment

def select_best_modelv0(experiment_name, metric="f1"):
    """
    Select best model overall
    """
    
    # get experiment by name
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)


    # Filter by run_name and tag `data_version` 
    filter_str = "run_name = 'Tuning' AND tags.data_version = 'v1'"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_str,
        order_by=["metrics.f1 DESC"],
    )

    # the first one will be the best F1
    best_run = runs[0]
    print("Best run ID:", best_run.info.run_id)
    print("Best F1:", best_run.data.metrics["f1"])


    # common part
    run_id = best_run.info.run_id

    # model
    model_uri = f"runs:/{run_id}/model"

    # encoder
    encoder_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="preprocessor/label_encoder.pkl"
    )

    model_type = best_run.data.tags.get("model_type")
    if model_type == 'nn':
        scaler_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="preprocessor/scaler.pkl"
        )
        # return
        return {
            "model_type": model_type,
            "model_uri": model_uri, 
            "scaler_path": scaler_path, 
            "encoder_path": encoder_path
        }

    else: # tree
        return {
            "model_type": model_type,
            "model_uri": model_uri, 
            "encoder_path": encoder_path
        }






"""
filter_str = "tags.data_version = 'v1.2' and tags.model_type = 'nn'"
"""



""" -> this corresponds to the saved val data
    if runs:
        best_run = runs[0]
        print("Best run ID:", best_run.info.run_id)
        
        # Get all metrics as a dict
        all_metrics = best_run.data.metrics
        print("All metrics for best run:")
        for metric_name, value in all_metrics.items():
            print(f"{metric_name}: {value}")
    else:
        print("No runs found!")
    """