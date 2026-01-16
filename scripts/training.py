import hydra
from omegaconf import DictConfig, OmegaConf
from src.preprocessors.factory import PreprocessorFactory
from src.training.factory import TrainerFactory
from src.inference.factory import PredictorFactory
import os
from src.utils.logging import logging
from src.utils.metrics import compute_metrics




@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # get model type (nn, tree.....)
    model_type = cfg.model_type
    print(f"\nSelected model: {model_type}")

    # Preprocessing
    preprocessor = PreprocessorFactory.get_preprocessor(model_type, cfg, cfg.preprocessor)
    X_train, X_val, y_train, y_val, artifacts = preprocessor.preprocess() # rename to clarify are preprocessed
    artifacts = preprocessor.get_artifacts()

    # Training
    trainer = TrainerFactory.get_trainer(
        model_type=model_type,
        num_classes=preprocessor.num_classes,
        average='weighted',
    )

    # config
    model_config = OmegaConf.load(f"config/models/{model_type}.yaml")
    train_config = OmegaConf.load(f"config/training/{model_type}.yaml")
    config = {
        "model": model_config,
        "training": train_config
    }
    
    # model_config, training_config...............
    results = trainer.train(X_train, y_train, X_val, y_val, config)





    # -------------- Predict ------------------------------
    # scaler and encoder
    scaler = artifacts.get("scaler") # None for trees
    encoder = artifacts.get("encoder")

    data_pred = {
            "model_type": model_type,
            "model": results.model,
            "scaler": scaler,
            "encoder": encoder,
            "device": "cpu"
    }

    predictor = PredictorFactory.get_predictor(**data_pred)

    preds = predictor.predict(X_val) # 0 0 1
    preds2 = predictor.predict_proba(X_val) # probs
    #preds3 = predictor.predict_logits(X_val) # logits
    preds4 = predictor.predict_labels(X_val)


    print(preds)
    print(preds2)
    print(preds4)

    val_metrics = compute_metrics(y_val, preds, 'weighted')

    print(results)
    print(type(results))
    print(val_metrics)


    from src.utils.results import Metrics
    #results.val.metrics = Metrics.from_dict(val_metrics)
    results.val.metrics = val_metrics


    # labels, preds and probs for cm and roc curve
    results.val.labels = y_val
    results.val.preds = preds
    results.val.probs = preds2

    



    # data for logging





    # Logging
    logging(cfg.experiment_name, 'Training', artifacts, results, model_type)





if __name__ == "__main__":
    main()










