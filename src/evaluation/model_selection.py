import joblib
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")



# nn_experiment

def select_best_model(experiment_name, metric="f1", higher_is_better=True):
    """
    Select best model overall
    """
    
    # get experiment by name
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)


    # Filter by tag `data_version`
    filter_str = "tags.data_version = 'v1'"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_str,
        order_by=["metrics.f1 DESC"],
    )

    # the first one will be the best F1
    best_run = runs[0]
    print("Best run ID:", best_run.info.run_id)
    print("Best F1:", best_run.data.metrics["f1"])


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


    run_id = best_run.info.run_id
    #run_id = "00cfc7deb01c4004b0e7bf1f38ef255e" # force a tree
    


    model_uri = f"runs:/{run_id}/model"



    print(best_run.data.tags.get("model_type"))


    try:
        scaler_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="preprocessor/scaler.pkl"
        )
    except:
        scaler_path = None



    encoder_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="preprocessor/label_encoder.pkl"
    )

    return {
            "model_type": best_run.data.tags.get("model_type", "unknown"),
            "model_uri": model_uri, 
            "scaler_path": scaler_path, 
            "encoder_path": encoder_path
        }




"""
filter_str = "tags.data_version = 'v1.2' and tags.model_type = 'nn'"
"""