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













# with config file for hidden dims


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.model(x)



# use SimpleNN



"""
Exactly — anything dynamic, derived from the data or experiment setup shouldn’t go into a static Hydra config. 
Configs are meant for fixed, user-defined parameters (like learning rate, batch size, model type).
"""


class nn_model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(nn_model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.model(x)
