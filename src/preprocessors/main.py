import hydra
from omegaconf import DictConfig
from src.preprocessors.factory import PreprocessorFactory




@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # get preprocessor    
    preprocessor = PreprocessorFactory.get_preprocessor(cfg, cfg.preprocessor) # e.g. NNPreprocessor

    # preprocess
    data = preprocessor.preprocess() 
    
    print(f"Data prepared for model type: {cfg.model_type}")
    
    if cfg.model_type.lower() in ['nn', 'neural', 'neuralnet']:
        train_loader, val_loader, artifacts = data # i might not need artifacts here if i use it like later
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    else:
        X_train, X_val, y_train, y_val, artifacts = data
        print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")


    print(preprocessor.get_artifacts())

    aux = preprocessor.get_artifacts()
    #print(aux['scaler'].mean_) only for nn



if __name__ == "__main__":
    main()



#python main.py model_type=tree do_smote=false smote_strategy=0.5
# python -m src.preprocessors.main