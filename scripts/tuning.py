import hydra
from omegaconf import DictConfig, OmegaConf
from src.preprocessors.factory import PreprocessorFactory
from src.tuning.factory import TunerFactory
import os
from src.utils.logging import logging


from src.training.factory import TrainerFactory


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):


    # get model type (nn, tree.....)
    model_type = cfg.model_type
    print(f"\nSelected model: {model_type}")

    # Preprocessing
    preprocessor = PreprocessorFactory.get_preprocessor(model_type, cfg, cfg.preprocessor)
    X_train, X_val, y_train, y_val, artifacts = preprocessor.preprocess()
    artifacts = preprocessor.get_artifacts()

    # Tuning
    cfg_tuning = OmegaConf.load(f"config/tuning/{model_type}.yaml") # use tuning/nn.yaml or tuning/tree.yaml....
    tuner = TunerFactory.get_tuner(
        model_type=model_type,
        cfg=cfg_tuning,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        preprocessor=preprocessor,
    ) # returns for ex NNTuner(cfg_tuning, X_train, y_train, X_val, y_val, num_classes) or TreeTuner...

    best_config = tuner.tune(num_samples=cfg.tuning.num_samples)

    # Train best model and get all metrics
    


    trainer = TrainerFactory.get_trainer(
        model_type=model_type,
        average='weighted',
        num_classes=preprocessor.num_classes,
    )

    


    # config
    print(best_config)
    # maybe merge what i tune with what i dont

    #results = trainer.train(X_train, y_train, X_val, y_val, best_config)

    # Logging
    #logging(cfg.experiment_name, 'Tuning', artifacts, results, model_type)





if __name__ == "__main__":
    main()




"""
python -m src.scripts.tuning
python -m src.scripts.tuning model_type=tree
python -m src.scripts.tuning -m model_type=nn,tree
"""


# see later // training
# python train.py -m models=tree,nn,rf
# python train.py -m models=tree,nn,rf hydra/launcher=submitit_local
# python train.py -m models=tree,nn lr=0.001,0.01 batch_size=32,64





