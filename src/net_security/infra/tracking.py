import mlflow
import mlflow.sklearn
import joblib
import os
import json
from mlflow.tracking import MlflowClient
from pathlib import Path





#--------------------select best model----------------------------
def _get_candidate_runs(
    experiment_name: str,
    metric: str,
    model_type: str | None,
    data_version: str,
):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # real project -> remember to filter by data

    # 1. Simplify the search filter to what MLflow can handle
    filter_str = f"tags.data_version = '{data_version}'"
    if model_type:
        filter_str += f" and tags.model_type = '{model_type}'"

    # 2. Get the runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_str,
        order_by=[f"metrics.{metric} DESC"],
    )

    # 3. Filter manually for the 'stage' tag
    candidate_runs = [
        run for run in runs 
        if run.data.tags.get("stage") in ["train", "tune"]
    ]

    if not candidate_runs:
        raise ValueError("No candidate runs found")

    return candidate_runs





def select_best_model(
    experiment_name: str,
    metric: str = "f1",
    model_type: str | None = None,
    data_version: str = "v1",
):
    runs = _get_candidate_runs(
        experiment_name,
        metric,
        model_type,
        data_version,
    )

    best = runs[0]
    run_id = best.info.run_id
    model_type = best.data.tags["model_type"]

    model_uri = f"runs:/{run_id}/model"

    encoder_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="preprocessor/label_encoder.pkl",
    )

    result = {
        "run_id": run_id,
        "model_type": model_type,
        "model_uri": model_uri,
        "encoder_path": encoder_path,
        "metric": best.data.metrics[metric],
    }

    if model_type == "nn":
        scaler_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="preprocessor/scaler.pkl",
        )
        result["scaler_path"] = scaler_path

    return result







# maybe later, select top




