import hydra
from omegaconf import DictConfig, OmegaConf
from src.preprocessors.factory import PreprocessorFactory
from src.training.factory import TrainerFactory
import os
from src.utils.logging import logging





@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):


    # get model type (nn, tree.....)
    model_type = cfg.model_type
    print(f"\nSelected model: {model_type}")

    # Preprocessing
    preprocessor = PreprocessorFactory.get_preprocessor(model_type, cfg, cfg.preprocessor)
    X_train, X_val, y_train, y_val, artifacts = preprocessor.preprocess()
    artifacts = preprocessor.get_artifacts()



    # Training
    #cfg_tuning = OmegaConf.load(f"config/tuning/{model_type}.yaml") # use tuning/nn.yaml or tuning/tree.yaml....
    trainer = TrainerFactory.get_trainer(
        model_type=model_type,
        average='weighted',
        num_classes=preprocessor.num_classes,
    )

    # Train best model and get all metrics
    # later pass appr. config
    config = {"hidden1": 64, "hidden2": 64, "lr":1e-3, "batch_size":32} # lr? bsize?


    config_tree = {"criterion":"log_loss", "max_depth":10, "min_samples_split":2}


    
    # model_config, training_config...............
    results = trainer.train(X_train, y_train, X_val, y_val, config_tree) #, 10) # 10 not applicabel for trees; wht goes heer it needs to cme from config; use **kwargs

    # criterion, max depth, min_sample split -> model_config
    #results = trainer.train(config)

    print(results)

    # Logging
    logging(cfg.experiment_name, 'Training', artifacts, results, model_type)





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





