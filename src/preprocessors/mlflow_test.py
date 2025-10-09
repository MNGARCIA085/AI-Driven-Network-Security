import hydra
from omegaconf import DictConfig
from src.preprocessors.factory import PreprocessorFactory
import mlflow



@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print(cfg)

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

    #aux = preprocessor.get_artifacts()
    #print(aux['scaler'].mean_) only for nn



    # testing MLFlow


    mlflow.set_experiment('test_prep')

    artifacts = preprocessor.get_artifacts()

    with mlflow.start_run(run_name='test'):
        # Log numeric and string metadata
        mlflow.log_param("balance_factor", artifacts["balance_factor"])
        mlflow.log_param("val_size", artifacts["val_size"])
        
        # Log class distributions as JSON
        mlflow.log_dict(artifacts["class_dist_before_smote"], "class_dist_before_smote.json")
        mlflow.log_dict(artifacts["class_dist_after_smote"], "class_dist_after_smote.json")
        
        # Optionally log shapes
        mlflow.log_param("train_shape", str(artifacts["train_shape"]))
        mlflow.log_param("val_shape", str(artifacts["val_shape"]))



if __name__ == "__main__":
    main()
