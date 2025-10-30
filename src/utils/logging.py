import mlflow
import mlflow.sklearn
from .plots import plot_cm, plot_roc, plot_train_val
import joblib
import os




# logging functions
def logging(artifacts, results, model_type):

    # Tags
    mlflow.set_tag("model_type", model_type)
    mlflow.set_tag("data_version", "v1") # hardcoded for now
    #mlflow.set_tag("data_version", artifacts["dvc_version"]) 


    # Common params
    mlflow.log_param("balance_factor", artifacts["balance_factor"])
    mlflow.log_param("val_size", artifacts["val_size"])
    mlflow.log_param("train_shape", str(artifacts["train_shape"]))
    mlflow.log_param("val_shape", str(artifacts["val_shape"]))    

    # Common artifacts
    mlflow.log_dict({"features": artifacts["features"]}, "features.json")
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


    # Optional artifacts (scaler only for NNs)
    if "scaler" in artifacts and artifacts["scaler"] is not None:
        filename = "scaler.pkl"
        joblib.dump(artifacts["scaler"], filename)
        mlflow.log_artifact(filename, artifact_path="preprocessor")
        os.remove(filename)

        # scaler type
        mlflow.log_param("scaler_type", artifacts["scaler_type"])


    # Metrics (shared)
    for m in ["accuracy", "precision", "recall", "f1"]:
        mlflow.log_metric(m, getattr(results.val.metrics, m))


    # Model-specific logs
    if model_type == "nn":
        # Model
        mlflow.pytorch.log_model(results.model, artifact_path="model")
        
        # training curves
        loss_path = plot_train_val(results.train.losses, results.val.losses, "loss_curve.png", 'Loss')
        mlflow.log_artifact(loss_path)
        os.remove(loss_path)


        acc_path = plot_train_val(results.train.accs, results.val.accs, "acc_curve.png", 'Accuracy')
        mlflow.log_artifact(acc_path)
        os.remove(acc_path)  
    
    else: # trees
        mlflow.sklearn.log_model(results.model, name="model")


    # hyperparams (differnt dict depending on the model)
    mlflow.log_params(results.hyperparams)


    # Common plots
    cm_path = plot_cm(results.val.labels, results.val.preds)
    mlflow.log_artifact(cm_path)
    os.remove(cm_path)

    roc_path = plot_roc(results.val.labels, results.val.probs)
    mlflow.log_artifact(roc_path)
    os.remove(roc_path)








# log test results
def log_test_results(model_type, results):
    with mlflow.start_run(run_name="test_evaluation"):

        # tags
        mlflow.set_tag("dataset", "test")
        mlflow.set_tag("data_version", "v1")

        mlflow.log_param("model_type", model_type)

        # Metrics (shared)
        for m in ["accuracy", "precision", "recall", "f1"]:
            mlflow.log_metric(m, getattr(results.metrics, m))

        # Plots
        cm_path = plot_cm(results.labels, results.preds)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        roc_path = plot_roc(results.labels, results.probs)
        mlflow.log_artifact(roc_path)
        os.remove(roc_path)




# analyze perfoemce when i move only one param






"""
# Delete the SQLite file
rm mlflow.db

# Delete all artifacts (by default in ./mlruns/)
rm -rf mlruns/

# Start MLflow server with SQLite backend
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000



# Start MLflow server with SQLite backend
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000



"""