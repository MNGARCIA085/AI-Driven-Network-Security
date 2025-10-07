import hydra
from omegaconf import DictConfig
from src.preprocessors.factory import PreprocessorFactory




@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    
    preprocessor = PreprocessorFactory.get_preprocessor(**cfg.preprocessor) #model_type=cfg.preprocessor.model_type, 
    

    data = preprocessor.preprocess()
    
    print(f"Data prepared for model type: {cfg.preprocessor.model_type}")
    
    if cfg.preprocessor.model_type.lower() in ['nn', 'neural', 'neuralnet']:
        train_loader, val_loader = data
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    else:
        X_train, X_val, y_train, y_val = data
        print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")

if __name__ == "__main__":
    main()



#python main.py model_type=tree do_smote=false smote_strategy=0.5
