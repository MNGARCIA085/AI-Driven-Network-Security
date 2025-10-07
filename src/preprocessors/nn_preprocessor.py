from .base import BasePreprocessor
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

class NNPreprocessor(BasePreprocessor):
    def __init__(self, batch_size=64, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.scaler = None

    def scale_features(self):
        scaler = StandardScaler()
        scaler.fit(self.X_train)

        # Scale training set
        self.X_train = pd.DataFrame(
            scaler.transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )

        # Scale validation set
        self.X_val = pd.DataFrame(
            scaler.transform(self.X_val),
            columns=self.X_val.columns,
            index=self.X_val.index
        )

        self.scaler = scaler
        return self

    def get_dataloaders(self):
        X_train_tensor = torch.tensor(self.X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train.values, dtype=torch.long)
        X_val_tensor = torch.tensor(self.X_val.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(self.y_val.values, dtype=torch.long)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def preprocess(self):
        self.load_data().encode_labels().split_features().apply_smote().scale_features()
        train_loader, val_loader = self.get_dataloaders()
        return train_loader, val_loader, self.get_artifacts()
