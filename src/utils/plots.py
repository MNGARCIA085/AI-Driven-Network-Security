import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np





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
    plt.plot(results["train_accs"], label="train_accuracy")
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
