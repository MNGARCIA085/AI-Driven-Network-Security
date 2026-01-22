from src.training.tree import TreeTrainer


def test_tree_trainer_train(dummy_data):
    X_train, y_train, X_val, y_val = dummy_data
    trainer = TreeTrainer(num_classes=2, average="binary")
    
    results = trainer.train(X_train, y_train, X_val, y_val, config={"model": {
        "criterion": "gini",
        "max_depth": 3,
        "min_samples_split": 2
    }})
    
    assert hasattr(results, "model")
    assert results.model is not None
    
    # check prediction works
    preds = results.model.predict(X_val)
    assert preds.shape[0] == X_val.shape[0]