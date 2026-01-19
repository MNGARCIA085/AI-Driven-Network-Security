import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import ray
from .base import BaseTuner
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from src.models.nnet import NNModel
from src.utils.metrics import compute_metrics
from src.utils.results import Results, Metrics
from .callbacks import EarlyStopping,LRReducer
from src.training.nn import NNTrainer



class NNTuner(BaseTuner):
    def __init__(self, cfg, X_train, y_train, X_val, y_val, num_classes): # see params later
        super().__init__(cfg, X_train, y_train, X_val, y_val) # parent class owns them
        
        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    # --- Ray train function ---
    @staticmethod
    def train_model_ray(config, X_train_id, y_train_id, X_val_id, y_val_id, num_classes):

        X_train = X_train_id # no need for X_train = ray.get(X_train_id); the call in the main module already desiarilize it
        y_train = y_train_id
        X_val = X_val_id
        y_val = y_val_id

        input_size = X_train.shape[1]

        model = NNModel(
            input_size=input_size,
            num_classes=num_classes,
            hidden1=config["model.hidden1"],
            hidden2=config["model.hidden2"]
        ) #.to(self.device) # not self in ray!!!

        optimizer = optim.Adam(model.parameters(), lr=config["training.lr"])
        criterion = nn.CrossEntropyLoss()
    
        # create a trainer
        trainer = NNTrainer(num_classes, 'weighted') # no hardcoding later!!!!!
        train_loader, val_loader = trainer.create_loaders(X_train, y_train, X_val, y_val,32)
        #------------------

        
        for _ in range(config["training.epochs"]):
            trainer.train_one_epoch(model, train_loader, optimizer, criterion)
            results = trainer.eval_one_epoch(model, val_loader, criterion)
            
            # Option 1: report just F1
            tune.report({"f1": results["f1"]})
            
            """
            # Option 2: report multiple metrics
            tune.report({
                "f1": results["f1"],
                "accuracy": results["accuracy"],
                "recall": results["recall"],
                "precision": results["precision"]
            })
            
            # Option 3: custom objective: e.g., weighted combination
            weighted_score = 0.7 * results["recall"] + 0.3 * results["f1"]
            tune.report({"weighted_score": weighted_score})
            """



    # Tuning config
    def get_tune_config(self):
        return {
            "model.hidden1": tune.choice(self.cfg.hidden1),
            "model.hidden2": tune.choice(self.cfg.hidden2),
            "training.batch_size": tune.choice(self.cfg.batch_size),
            "training.lr": tune.loguniform(self.cfg.lr.min, self.cfg.lr.max),
            "training.epochs": 5 # later add epcosh_trials too
        }








