from .base import BaseTuningConfig
from .nn import build_nn_tuning_config
from .tree import build_tree_tuning_config


def build_tuning_config(model_type: str, cfg: dict) -> BaseTuningConfig:
    if model_type == "nn":
        return build_nn_tuning_config(cfg)

    if model_type == "tree":
        return build_tree_tuning_config(cfg)

    raise ValueError(f"Unknown model_type: {model_type}")