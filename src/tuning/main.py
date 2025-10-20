from .nn import NNTuner
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from src.preprocessors.factory import PreprocessorFactory
import os
from src.utils.logging import logging



mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set default folder for artifacts
artifact_dir = os.path.abspath("./mlruns")  # choose any folder
os.makedirs(artifact_dir, exist_ok=True)

mlflow.set_experiment("nn_experiment")  # ensures artifact path is set




@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # add logging

    # get data from prep
    model_type = cfg.model_type # # model_name = cfg.models.name, maybe is better there?
    print(f"\nSelected model: {model_type}")


    with mlflow.start_run(run_name='Tuning'):

        # -------------------------------
        #  Preprocess data
        # -------------------------------
        preprocessor = PreprocessorFactory.get_preprocessor(model_type, cfg, cfg.preprocessor)
        X_train, X_val, y_train, y_val, artifacts = preprocessor.preprocess()
        print(f"Training data shape: {X_train.shape}")

        # number of classes, maybe it should be in num_classes
        num_classes = preprocessor.num_classes

        artifacts = preprocessor.get_artifacts()
        # Log numeric and string metadata
        
        # Tune
        cfg_tuning = OmegaConf.load(f"config/tuning/{model_type}.yaml") #use tuning/nn.yaml
        tuner = NNTuner(cfg_tuning, X_train, y_train, X_val, y_val, num_classes)
        best_config = tuner.tune(num_samples=1) # 10

        # Train best model and get all metrics
        results = tuner.train_best_model(best_config)


        # logging        
        logging(artifacts, results, 'nn') # then model_type


if __name__ == "__main__":
    main()



#-------https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html ; # as input true labesl and pred scores
# desconfiar!!!! roc too good