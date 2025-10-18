from .nn import NNTuner
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf


from src.preprocessors.factory import PreprocessorFactory

import matplotlib.pyplot as plt
import os
import joblib


import numpy as np


from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

from sklearn.preprocessing import label_binarize





mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set default folder for artifacts
artifact_dir = os.path.abspath("./mlruns")  # choose any folder
os.makedirs(artifact_dir, exist_ok=True)


mlflow.set_experiment("nn_experiment")  # ensures artifact path is set










#


def plot_loss(results):
    plt.figure()
    plt.plot(results["train_losses"], label="train_loss")
    plt.plot(results["val_losses"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    train_val_loss_path = "loss_curve.png"
    plt.savefig(train_val_loss_path)
    plt.close()
    return train_val_loss_path
    


def plot_acc(results):
    # Validation accuracy curve
    plt.figure()
    plt.plot(results["val_accs"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    val_acc_path = "val_acc_curve.png"
    plt.savefig(val_acc_path)
    plt.close()
    return val_acc_path
    



def plot_cm(results):
    cm = confusion_matrix(results["val_labels"], results["val_preds"])
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    return cm_path




def plot_roc(results):
    y_true = np.array(results["val_labels"])        # shape (n_samples,)
    y_score = np.array(results["val_preds_proba"])  # shape (n_samples, n_classes)
    num_classes = y_score.shape[1]

    # Binarize labels for multiclass One-vs-Rest ROC
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    plt.figure()
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right")

    # Save figure and log to MLflow
    roc_path = "roc_multiclass.png"
    plt.savefig(roc_path)
    plt.close()

    return roc_path




# -> encoder!!!!!!; features!!!!!!!!!!!!!!!!


# logging functions
def logging(artifacts, results):
    # ------------Params----------------------------#
    mlflow.log_param("balance_factor", artifacts["balance_factor"])
    mlflow.log_param("val_size", artifacts["val_size"])
    mlflow.log_param("train_shape", str(artifacts["train_shape"]))
    mlflow.log_param("val_shape", str(artifacts["val_shape"]))
    


    # ------------Artifacts---------------------------#
    # Log class distributions as JSON
    mlflow.log_dict(artifacts["class_dist_before_smote"], "class_dist_before_smote.json")
    mlflow.log_dict(artifacts["class_dist_after_smote"], "class_dist_after_smote.json")
    # scaler    
    scaler = artifacts["scaler"]
    filename = "scaler.pkl"
    joblib.dump(scaler, filename)
    mlflow.log_artifact(filename, artifact_path="preprocessor")
    os.remove(filename)


    # ------------Metrics---------------------------#
    mlflow.log_metric("accuracy", results["accuracy"])
    mlflow.log_metric("precision", results["precision"])
    mlflow.log_metric("recall", results["recall"])
    mlflow.log_metric("f1", results["f1"])

    # --- Log the model ----------------------------#
    mlflow.pytorch.log_model(results["model"], artifact_path="model") # it is saved in models

    # -----------------Plots----------------#
    train_val_loss_path = plot_loss(results)
    mlflow.log_artifact(train_val_loss_path)
    os.remove(train_val_loss_path)


    val_acc_path = plot_acc(results)
    mlflow.log_artifact(val_acc_path)
    os.remove(val_acc_path)

    cm_path = plot_cm(results)
    mlflow.log_artifact(cm_path)
    os.remove(cm_path)

    roc_path = plot_roc(results)
    mlflow.log_artifact(roc_path)
    os.remove(roc_path)


    #
    mlflow.log_dict({"features": artifacts["features"]}, "features.json")

    encoder = artifacts["encoder"]
    filename = "label_encoder.pkl"
    joblib.dump(encoder, filename)
    mlflow.log_artifact(filename, artifact_path="preprocessor")
    os.remove(filename)

    mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    mlflow.log_dict(mapping, "encoder_mapping.json")







@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # add logging

    # get data from prep
    # model_name = cfg.models.name
    model_name = cfg.model_type
    print(f"\nSelected model: {model_name}")


    with mlflow.start_run(run_name='Tuning'):

        # -------------------------------
        # 3️⃣ Preprocess data
        # -------------------------------
        preprocessor = PreprocessorFactory.get_preprocessor(model_name, cfg, cfg.preprocessor)
        X_train, X_val, y_train, y_val, artifacts = preprocessor.preprocess()
        print(f"Training data shape: {X_train.shape}")

        #input_size = preprocessor.input_size
        num_classes = preprocessor.num_classes

        artifacts = preprocessor.get_artifacts()
        # Log numeric and string metadata
        
        # Tune
        tuner = NNTuner(cfg, X_train, y_train, X_val, y_val, num_classes)
        best_config = tuner.tune(num_samples=10)

        # Train best model and get all metrics
        results = tuner.train_best_model(best_config)


        # logging        
        logging(artifacts, results)


if __name__ == "__main__":
    main()



#-------https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html ; # as input true labesl and pred scores
# desconfiar!!!! roc too good