
from .tree import TreeTrainer
from .nn import NNTrainer



class TrainerFactory:
    @staticmethod
    def get_trainer(model_name, cfg, model_cfg, input_size, num_classes):
        if model_name == "tree" or model_name == "rf":
            return TreeTrainer(cfg, model_cfg, input_size, num_classes)
        elif model_name == "nn":
            return NNTrainer(cfg, model_cfg, input_size, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")


"""
later -> trees dont need input size and num_classes
# src/training/factory.py
from src.training.nn_trainer import NNTrainer
from src.training.tree_trainer import TreeTrainer
# import other trainers as needed

class TrainerFactory:
    @staticmethod
    def get_trainer(model_name, cfg, model_cfg, preprocessor=None):
        
        Returns an instantiated trainer for the given model.
        If the trainer needs input_size and num_classes (like NNTrainer),
        it can get it from the preprocessor.
        
        if model_name.lower() == "nn":
            if preprocessor is None:
                raise ValueError("Preprocessor required for NNTrainer to get input_size and num_classes")
            return NNTrainer(
                input_size=preprocessor.input_size,
                num_classes=preprocessor.num_classes,
                cfg=cfg,
                model_cfg=model_cfg
            )
        elif model_name.lower() == "tree":
            return TreeTrainer(cfg, model_cfg)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")


trainer = TrainerFactory.get_trainer(
    model_name,
    cfg,
    cfg.models,
    preprocessor=preprocessor  # <-- needed for NNTrainer
)

trainer = NNTrainer(
    input_size=preprocessor.input_size,
    num_classes=preprocessor.num_classes,
    cfg=cfg,
    model_cfg=cfg.models
)



"""