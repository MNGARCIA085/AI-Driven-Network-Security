from net_security.training.nn import NNTrainer


def test_nn_trainer_train(dummy_data):
    X_train, y_train, X_val, y_val = dummy_data
    trainer = NNTrainer(num_classes=2, average="weighted")
    
    config = {
        "model": {"hidden1": 8, "hidden2": 4},
        "training": {"epochs": 2, "batch_size": 4, "lr": 1e-2}
    }
    results = trainer.train(X_train, y_train, X_val, y_val, config=config)
    
    assert hasattr(results, "model")
    assert results.model is not None
    
    # check that some metrics exist
    assert "losses" in results.train.__dict__
    assert "accs" in results.val.__dict__