import pytest
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


from src.training.nn import NNTrainer
from src.training.tree import TreeTrainer
from src.inference.nn import NNPredictor
from src.inference.tree import TreePredictor
from src.evaluation.base import Evaluator



# ------------------------
# Integration test for full pipeline
# ------------------------
def test_full_pipeline(dummy_data, dummy_encoder):
    X_train, y_train, X_val, y_val = dummy_data

    # --- NN pipeline ---
    nn_trainer = NNTrainer(num_classes=2, average="binary")
    nn_config = {
        "model": {"hidden1": 4, "hidden2": 2},
        "training": {"epochs": 1, "batch_size": 2, "lr": 1e-2}
    }
    nn_results = nn_trainer.train(X_train, y_train, X_val, y_val, nn_config)
    assert nn_results.model is not None

    nn_predictor = NNPredictor(nn_results.model, dummy_encoder, device="cpu")
    nn_preds = nn_predictor.predict(X_val)
    nn_labels = nn_predictor.predict_labels(X_val)
    assert nn_preds.shape[0] == X_val.shape[0]
    assert nn_labels.shape[0] == X_val.shape[0]

    nn_eval = Evaluator(average="binary")
    nn_metrics = nn_eval.compute_metrics(y_val, nn_preds)
    assert "accuracy" in nn_metrics and "f1" in nn_metrics

    # --- Tree pipeline ---
    tree_trainer = TreeTrainer(num_classes=2, average="binary")
    tree_config = {"model": {"criterion": "gini", "min_samples_split": 2}}
    tree_results = tree_trainer.train(X_train, y_train, X_val, y_val, tree_config)
    assert tree_results.model is not None

    tree_predictor = TreePredictor(tree_results.model, dummy_encoder)
    tree_preds = tree_predictor.predict(X_val)
    tree_labels = tree_predictor.predict_labels(X_val)
    assert tree_preds.shape[0] == X_val.shape[0]
    assert tree_labels.shape[0] == X_val.shape[0]

    tree_eval = Evaluator(average="binary")
    tree_metrics = tree_eval.compute_metrics(y_val, tree_preds)
    assert "accuracy" in tree_metrics and "f1" in tree_metrics
