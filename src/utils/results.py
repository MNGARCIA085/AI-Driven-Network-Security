# utils/results.py
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class Metrics:
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    avg_loss: Optional[float] = None # new; only for NNs


@dataclass
class StageResults:
    losses: Optional[List[float]] = None
    accs: Optional[List[float]] = None
    preds: Optional[List[Any]] = None
    labels: Optional[List[Any]] = None
    probs: Optional[List[Any]] = None
    #metrics: Metrics = field(default_factory=Metrics)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


@dataclass
class Results:
    train: StageResults = field(default_factory=StageResults)
    val: StageResults = field(default_factory=StageResults)
    model: Optional[Any] = None
    hyperparams: Optional[Dict] = None

    def to_dict(self):
        return {
            "train": {
                "losses": self.train.losses,
                "accs": self.train.accs,
                "metrics": vars(self.train.metrics),
            },
            "val": {
                "losses": self.val.losses,
                "accs": self.val.accs,
                "preds": self.val.preds,
                "labels": self.val.labels,
                "probs": self.val.probs,
                "metrics": vars(self.val.metrics),
            },
            "model": self.model,
            "hyperarams": self.hyperparams,
        }


@dataclass
class TestResults:
    preds: List[Any]
    labels: List[Any]
    probs: List[Any]
    metrics: Metrics = field(default_factory=Metrics)







