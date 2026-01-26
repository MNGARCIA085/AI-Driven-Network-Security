import os
import hydra
from omegaconf import DictConfig, OmegaConf

from net_security.preprocessors.factory import PreprocessorFactory
from net_security.training.factory import TrainerFactory
from net_security.inference.factory import PredictorFactory
from net_security.evaluation.base import Evaluator
from net_security.infra.config import init_mlflow
from net_security.infra.logging import logging
from net_security.preprocessors.config import build_data_config




@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # init mlflow
    init_mlflow(cfg.experiment_name)

    # get model type (nn, tree.....)
    model_type = cfg.model_type.name
    print(f"\nSelected model: {model_type}")

    # preprocessing
    data_cfg = build_data_config(cfg.preprocessor, cfg)
    preprocessor = PreprocessorFactory.get_preprocessor(model_type,data_cfg)
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
    # Get encoder
    encoder = artifacts.get("encoder")

    data_pred = {
            "model_type": model_type,
            "model": results.model,
            "encoder": encoder,
            "device": "cpu"
    }

    # Predictor
    predictor = PredictorFactory.get_predictor(**data_pred)

    preds = predictor.predict(X_val) # 0 0 1
    probs = predictor.predict_proba(X_val)
    preds_labels = predictor.predict_labels(X_val) # DDos, BENIGN....


    #---------------Evaluation----------------------------
    evaluator = Evaluator() # include average later appr.
    val_metrics = evaluator.compute_metrics(y_val, preds)


    #-------------Results--------------------------------
    results.val.metrics = val_metrics
    # labels, preds and probs for cm and roc curve
    results.val.labels = y_val
    results.val.preds = preds
    results.val.probs = probs


    #-------------Logging --------------------------------
    logging('Training', artifacts, results, model_type, 'train')





if __name__ == "__main__":
    main()



"""
python -m scripts.training -m model_type=nn,tree preprocessor.val_size=.4,.6
"""






