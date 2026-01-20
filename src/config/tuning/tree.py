from dataclasses import dataclass
from typing import List
from .base import BaseTuningConfig


#---------Dataclass-----------------------------
@dataclass(frozen=True)
class IntRange:
    min: int
    max: int


@dataclass(frozen=True)
class TreeTuningConfig(BaseTuningConfig):
    criterion: List[str]
    max_depth: IntRange
    min_samples_split: IntRange



#---------------Builder-------------------------
def build_tree_tuning_config(cfg: dict) -> TreeTuningConfig:
    return TreeTuningConfig(
        num_samples=cfg["num_samples"],
        average=cfg["average"],
        criterion=cfg["criterion"],
        max_depth=IntRange(
            min=cfg["max_depth"]["min"],
            max=cfg["max_depth"]["max"],
        ),
        min_samples_split=IntRange(
            min=cfg["min_samples_split"]["min"],
            max=cfg["min_samples_split"]["max"],
        ),
    )