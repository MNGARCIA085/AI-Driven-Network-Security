import torch
import torch.nn as nn


class NNModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden1=128, hidden2=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.model(x)

