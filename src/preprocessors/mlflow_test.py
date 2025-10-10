import hydra
from omegaconf import DictConfig
from src.preprocessors.factory import PreprocessorFactory
import mlflow



# Set tracking URI (SQLite)
mlflow.set_tracking_uri("sqlite:///mlflow.db")


#mlflow ui --backend-store-uri sqlite:///mlflow.db


"""

By default, MLflow stores artifacts (models, plots, etc.) in a local folder (mlruns/).
You can change this to another path or a cloud storage:

mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns


save artfs ina  cloud storage:
# Artifact location can be on S3
mlflow.set_artifact_uri("s3://your-bucket/mlflow-artifacts")

"""




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




    # scaler login!!!!


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


        # also save scaler
        """  
        import joblib
        scaler = artifacts["scaler"]
        joblib.dump(scaler, "scaler.pkl")
        mlflow.log_artifact("scaler.pkl", artifact_path="preprocessor")

        # i shold remove it later for clarity
        """
        import joblib
        import os
        scaler = artifacts["scaler"]
        filename = "scaler.pkl"

        # Save + log + clean up
        joblib.dump(scaler, filename)
        mlflow.log_artifact(filename, artifact_path="preprocessor")
        os.remove(filename)



if __name__ == "__main__":
    main()


# maybe include mlruns in thios so i have artifacts!!!!