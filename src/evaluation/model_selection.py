import joblib
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")



# nn_experiment

def select_best_model(experiment_name, metric="f1"):
    """
    Select best model overall
    """
    
    # get experiment by name
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)


    # Filter by tag `data_version`
    #filter_str = "tags.data_version = 'v1'"
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
    #run_id = "00cfc7deb01c4004b0e7bf1f38ef255e" # force a tree; force an nn to test
    #run_id = "6faf249e0eac4db98156628c09fccb9a" # force an NN


    # model
    model_uri = f"runs:/{run_id}/model"

    # encoder
    encoder_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="preprocessor/label_encoder.pkl"
    )

    model_type = best_run.data.tags.get("model_type")
    #model_type = 'nn' # force nn
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