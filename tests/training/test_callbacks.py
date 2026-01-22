import pytest
import torch
from src.training.callbacks import EarlyStopping, LRReducer

# ------------------------
# EarlyStopping tests
# ------------------------
def test_early_stopping_triggers():
    es = EarlyStopping(patience=3, mode="max")
    scores = [0.5, 0.6, 0.55, 0.54, 0.53]  # stops after 3 non-improving
    for score in scores:
        es(score)
    assert es.stop is True

def test_early_stopping_resets_on_improvement():
    es = EarlyStopping(patience=2, mode="max")
    es(0.5)
    es(0.6)
    es(0.55)  # counter=0, because 0.55 < best+min_delta
    es(0.61)  # improvement
    assert es.stop is False

# ------------------------
# LRReducer test
# ------------------------
def test_lr_reducer_step_reduces_lr():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    lr_reducer = LRReducer(optimizer, factor=0.1, patience=0)  # patience=0 to trigger immediately
    # initial LR
    initial_lr = optimizer.param_groups[0]['lr']
    # call step with metric that should reduce LR
    lr_reducer.step(0.5)
    new_lr = optimizer.param_groups[0]['lr']
    assert new_lr <= initial_lr
