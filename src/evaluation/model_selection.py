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

    # get all runs in the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1 DESC"],  # sort by F1 descending
    )

    # the first one will be the best F1
    best_run = runs[0]
    print("Best run ID:", best_run.info.run_id)
    print("Best F1:", best_run.data.metrics["f1"])

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


    run_id = best_run.info.run_id
    run_id = "c63f08b6f1c542ee97ea16aba64cba46" # for now, make sure is a NN
    model_uri = f"runs:/{run_id}/model"
    # loaded_model = mlflow.pytorch.load_model(model_uri)

    scaler_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="preprocessor/scaler.pkl"
    )
    # scaler = joblib.load(local_path); it doiesnbt exist for the trees

    encoder_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="preprocessor/label_encoder.pkl"
    )

    return model_uri, scaler_path, encoder_path
