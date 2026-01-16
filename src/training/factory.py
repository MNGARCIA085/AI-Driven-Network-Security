from .tree import TreeTrainer
from .nn import NNTrainer




# preprocessor.num_classes
# maybe pass data from a config dict

class TrainerFactory:
    @staticmethod
    def get_trainer(model_type, num_classes, average):
        if model_type.lower() == "tree":
            return TreeTrainer(num_classes, average)
        elif model_type.lower() == "nn":
            return NNTrainer(num_classes, average)
        else:
            raise ValueError(f"Unknown tuner type: {model_type}")
