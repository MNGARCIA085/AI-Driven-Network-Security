# utils/results.py
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class Metrics:
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None


@dataclass
class StageResults:
    losses: Optional[List[float]] = None
    accs: Optional[List[float]] = None
    preds: Optional[List[Any]] = None
    labels: Optional[List[Any]] = None
    probs: Optional[List[Any]] = None
    metrics: Metrics = field(default_factory=Metrics)


@dataclass
class Results:
    train: StageResults = field(default_factory=StageResults)
    val: StageResults = field(default_factory=StageResults)
    model: Optional[Any] = None

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
        }


"""
results = Results()
results.val.preds = val_preds
results.val.labels = val_labels
results.val.probs = val_probs
results.val.metrics = Metrics(acc=val_acc, f1=val_f1)
results.model = model
return results.to_dict() # if needed
"""

"""
for test
@dataclass
class TestResults:
    preds: List[Any]
    labels: List[Any]
    probs: List[Any]
    metrics: Metrics = field(default_factory=Metrics)
"""
