import mlflow
import mlflow.sklearn
import joblib
import os
import json

from dataclasses import is_dataclass, asdict
from net_security.utils.plots import plot_cm, plot_roc, plot_train_val

from hydra.core.hydra_config import HydraConfig


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
    mlflow.log_dict(artifacts["class_dist_before_smote"], "class_dist_before_smote.json")
    mlflow.log_dict(artifacts["class_dist_after_smote"], "class_dist_after_smote.json")


    # encoder

    #out_dir = HydraConfig.get().runtime.output_dir
    #save_path = os.path.join(out_dir, plot_name)


    encoder = artifacts["encoder"]
    
    #filename = "label_encoder.pkl"
    out_dir = HydraConfig.get().runtime.output_dir
    filename = os.path.join(out_dir, "label_encoder.pkl")
    

    joblib.dump(encoder, filename)
    mlflow.log_artifact(filename, artifact_path="preprocessor")
    os.remove(filename)
    mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    mlflow.log_dict(mapping, "encoder_mapping.json")


    # scaler (only for NNs)
    if artifacts.get("scaler") is not None:
        
        #filename = "scaler.pkl"
        

        out_dir = HydraConfig.get().runtime.output_dir
        filename = os.path.join(out_dir, "scaler.pkl")

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
def logging(run_name, artifacts, results, model_type, stage):

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
def log_test_results(tuning_run_id, model_type, results, stage='eval'):
    with mlflow.start_run(run_name="test_evaluation"):
        log_tags(stage, model_type, "test", {"tuning_run_id": tuning_run_id})
        log_metrics(results.metrics)
        log_plots(results, model_type)
