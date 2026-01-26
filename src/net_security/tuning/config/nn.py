from dataclasses import dataclass
from typing import List
from .base import BaseTuningConfig


#----------Dataclass-----------------------
@dataclass(frozen=True)
class LogUniform:
    min: float
    max: float

# SearchSpace = Union[List[int], LogUniform]


@dataclass(frozen=True)
class NNTuningConfig(BaseTuningConfig):
    hidden1: List[int]
    hidden2: List[int]
    batch_size: List[int]
    lr: LogUniform
    epochs_trials: int


#----------Builder------------------------------
def build_nn_tuning_config(cfg: dict) -> NNTuningConfig:
    return NNTuningConfig(
        num_samples=cfg["num_samples"],
        average=cfg["average"],
        epochs_trials=cfg["epochs_trials"],
        hidden1=cfg["hidden1"],
        hidden2=cfg["hidden2"],
        batch_size=cfg["batch_size"],
        lr=LogUniform(
            min=cfg["lr"]["min"],
            max=cfg["lr"]["max"],
        ),
    )