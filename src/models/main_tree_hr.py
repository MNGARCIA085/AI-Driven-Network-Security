import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import classification_report, confusion_matrix

# Adjust imports to your project
# from your_package.preprocessor import PreprocessorFactory
# from your_package.models import tree_model, nn_model, rf_model





@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print("\n=== Full Config ===")
    print(OmegaConf.to_yaml(cfg))

    # -----------------------------------------------------------
    # 1️⃣ Determine the model name
    # -----------------------------------------------------------
    #model_name = cfg.models.name
    model_name = cfg.model_type
    print(f"\nSelected model: {model_name}")



    """
    # -----------------------------------------------------------
    # 2️⃣ Preprocess data
    # -----------------------------------------------------------
    preprocessor = PreprocessorFactory.get_preprocessor(model_name, cfg, cfg.preprocessor)
    X_train, X_val, y_train, y_val, artifacts = preprocessor.preprocess()
    print(f"\nTraining data shape: {X_train.shape}")

    # -----------------------------------------------------------
    # 3️⃣ Initialize the correct model
    # -----------------------------------------------------------
    if model_name == "tree":
        model = tree_model(cfg, cfg.models)
    elif model_name == "nn":
        model = nn_model(cfg, cfg.models)
    elif model_name == "rf":
        model = rf_model(cfg, cfg.models)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # -----------------------------------------------------------
    # 4️⃣ Train the model
    # -----------------------------------------------------------
    model.fit(X_train, y_train)

    # -----------------------------------------------------------
    # 5️⃣ Evaluate
    # -----------------------------------------------------------
    y_pred = model.predict(X_val)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=3))
    """


if __name__ == "__main__":
    main()


#python -m src.models.main_tree_hr model_type=tree
#python -m src.models.main_tree_hr -m model_type=tree,nn  (multiruns)
# hace 4: python -m src.models.main_tree_hr -m model_type=tree,nn batch_size=32,64