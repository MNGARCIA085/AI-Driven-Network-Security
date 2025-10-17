import mlflow
from mlflow.tracking import MlflowClient

# connect to your tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()




def main():
    
    # get experiment by name
    experiment = client.get_experiment_by_name("nn_experiment")

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

    """
    best_run.data.params   # dictionary of all logged parameters
    best_run.data.tags     # dictionary of all tags
    """


if __name__=="__main__":
    main()


"""
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

experiment = client.get_experiment_by_name("nn_experiment")

# Suppose your tag key is "data_hash" and value is "abc123"
filter_str = "tags.data_hash = 'abc123'"

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=filter_str,
    order_by=["metrics.f1 DESC"]  # sort by F1 descending
)

if runs:
    best_run = runs[0]
    print("Best run ID:", best_run.info.run_id)
    print("Best F1:", best_run.data.metrics["f1"])
else:
    print("No runs found with this tag!")


filter_str = "tags.data_hash = 'abc123' AND tags.model_type = 'NN'"


"""