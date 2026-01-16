import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return
        if (self.mode == "max" and current_score > self.best_score + self.min_delta) or \
           (self.mode == "min" and current_score < self.best_score - self.min_delta):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


class LRReducer:
    def __init__(self, optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr)

    def step(self, metric):
        self.scheduler.step(metric)
