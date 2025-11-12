import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import os

from hydra.utils import to_absolute_path


def plot_train_val(train_values, val_values, plot_name, ylabel, xlabel="Epoch"):
    """
    Plot training and validation curves (it can be for loss or accs)
    
    Args:
        train_values (list or array): Training values per epoch.
        val_values (list or array): Validation values per epoch.
        ylabel (str): Label for y-axis
        xlabel (str): Label for x-axis (default "Epoch").
        plot_name (str): Acc or Loss.
        
    Returns:
        str: The path where the plot was saved.
    """
    plt.figure()
    plt.plot(train_values, label=f"Train {ylabel}")
    plt.plot(val_values, label=f"Validation {ylabel}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(os.getcwd(), plot_name)
    plt.savefig(save_path)
    plt.close()
    return save_path





def plot_cm(labels, preds):
    """
    Plot and save a confusion matrix for classification results.

    Args:
        labels (array-like): True class labels.
        preds (array-like): Predicted class labels by the model.

    Returns:
        str: The file path where the confusion matrix image was saved.
    """
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(os.getcwd(), "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    return cm_path




def plot_roc(labels, probs):
    """
    Plot and save ROC curves for a multiclass classification task using one-vs-rest approach.

    Args:
        labels (array-like): True class labels (integers from 0 to n_classes-1).
        probs (array-like): Predicted probabilities with shape (n_samples, n_classes).
        class_names (list, optional): List of class names for labeling the curves. Defaults to None.

    Returns:
        str: The file path where the ROC curves image was saved.
    """
    y_true = np.array(labels)  # shape (n_samples,)
    y_score = np.array(probs)  # shape (n_samples, n_classes)
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
    #roc_path = "roc_multiclass.png"
    roc_path = os.path.join(os.getcwd(), "roc_multiclass.png") # save in appropiate output dir so i dont have problems with // exec
    plt.savefig(roc_path)
    plt.close()

    return roc_path
