from .tree import TreeTrainer
from .nn import NNTrainer




# preprocessor.num_classes
# maybe pass data from a config dict

class TrainerFactory:
    @staticmethod
    def get_trainer(model_type, average, num_classes):
        if model_type.lower() == "tree":
            return TreeTrainer(average, num_classes)
        elif model_type.lower() == "nn":
            return NNTrainer(average, num_classes)
        else:
            raise ValueError(f"Unknown tuner type: {model_type}")
