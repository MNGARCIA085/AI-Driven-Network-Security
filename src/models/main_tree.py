import torch
import torch.nn as nn
import torch.optim as optim
from src.models.nnet import SimpleNN
import hydra
from omegaconf import DictConfig
from src.preprocessors.factory import PreprocessorFactory


from .tree import tree_model



from hydra import initialize, compose,  initialize_config_dir
from omegaconf import OmegaConf


from sklearn.metrics import classification_report, confusion_matrix


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):


    # get appropiate preprocessor
    preprocessor = PreprocessorFactory.get_preprocessor('tree', cfg, cfg.preprocessor)
    #print(preprocessor)

    #data = preprocessor.preprocess()

    X_train, X_val, y_train, y_val, artifacts = preprocessor.preprocess()

    print(X_train)


    # appropiate file for tree conf.
    # Load additional config file (e.g., tree.yaml inside config/models)

    model_cfg = hydra.compose(config_name="models/tree")

    print("\nTree model config:")
    print(OmegaConf.to_yaml(model_cfg))



    tree = tree_model(cfg, model_cfg.models)
    tree.fit(X_train, y_train)

    # Evaluate
    y_pred = tree.predict(X_val)

    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred, digits=3))




if __name__==main():
    main()