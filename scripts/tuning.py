import os
import hydra
from omegaconf import DictConfig, OmegaConf
from src.preprocessors.factory import PreprocessorFactory
from src.tuning.factory import TunerFactory
from src.utils.logging import logging
from src.training.factory import TrainerFactory
from src.utils.config_utils import unflatten_config
from src.inference.factory import PredictorFactory
from src.evaluation.base import Evaluator
from src.config.data import build_data_config
from src.config.tuning.factory import build_tuning_config



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):


    # get model type (nn, tree.....)
    model_type = cfg.model_type
    print(f"\nSelected model: {model_type}")

    # -------------- 1. Preprocessing ----------------------
    data_cfg = build_data_config(cfg.preprocessor, cfg)
    preprocessor = PreprocessorFactory.get_preprocessor(model_type,data_cfg)
    X_train, X_val, y_train, y_val, artifacts = preprocessor.preprocess()
    artifacts = preprocessor.get_artifacts()

    # ----------- 2. Tuning--------------------------
    """
    cfg_tuning = OmegaConf.load(f"config/tuning/{model_type}.yaml") # 
        uses tuning/nn.yaml or tuning/tree.yaml....
        hydra does not overrides this
        not ideal
    """
    
    tuning_cfg = build_tuning_config(model_type, cfg.tuning) # raw_cfg
    print(tuning_cfg)

    

    tuner = TunerFactory.get_tuner(
        model_type=model_type,
        cfg=tuning_cfg,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        preprocessor=preprocessor,
    ) # returns for ex NNTuner(cfg_tuning, X_train, y_train, X_val, y_val, num_classes) or TreeTuner...

    best_config = tuner.tune()
    print(best_config)
    config = unflatten_config(best_config)

    # -------- 3. Train the model with the best config ------------------
    # Trainer
    trainer = TrainerFactory.get_trainer(
        model_type=model_type,
        average='weighted',
        num_classes=preprocessor.num_classes,
    )

    results = trainer.train(X_train, y_train, X_val, y_val, config)
    print(results)


    # ---------- 4. Assemble results-------------------------
    encoder = artifacts.get("encoder")

    data_pred = {
            "model_type": model_type,
            "model": results.model,
            "encoder": encoder,
            "device": "cpu"
    }

    # preds
    predictor = PredictorFactory.get_predictor(**data_pred)
    preds = predictor.predict(X_val) # 0 0 1
    probs = predictor.predict_proba(X_val)
    preds_labels = predictor.predict_labels(X_val)


    # Val metrics
    evaluator = Evaluator() # include average later appr.
    val_metrics = evaluator.compute_metrics(y_val, preds)

    # add to results
    results.val.metrics = val_metrics
    results.val.labels = y_val
    results.val.preds = preds
    results.val.probs = probs


    # ---------------- 5. Logging --------------------
    logging(cfg.experiment_name, 'Tuning', artifacts, results, model_type)





if __name__ == "__main__":
    main()



"""
REAL EXEC:


python -m scripts.tuning tuning=nn
python -m scripts.tuning tuning=tree

-> python -m scripts.tuning model_type=tree tuning=tree tuning.num_samples=4

"""







# see later // training
# python train.py -m models=tree,nn,rf
# python train.py -m models=tree,nn,rf hydra/launcher=submitit_local
# python train.py -m models=tree,nn lr=0.001,0.01 batch_size=32,64





