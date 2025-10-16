from .base import BasePreprocessor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd







class NNPreprocessor(BasePreprocessor):
    def __init__(self, global_cfg, pre_cfg):
        super().__init__(global_cfg, pre_cfg)
        # NN-specific scaler type from pre_cfg
        #self.scaler_type = getattr(pre_cfg, "scaler_type", "standard")


    def scale_features(self):
        if self.scaler_type == "none":
            return self

        # select scaler type
        scaler_map = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
            "normalize": Normalizer,
        }

        scaler_cls = scaler_map.get(self.scaler_type)
        if scaler_cls is None:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        self.scaler = scaler_cls()

        # fit (only with training data)        
        self.scaler.fit(self.X_train)

        # Scale training set
        self.X_train = pd.DataFrame(
            self.scaler.transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )

        # Scale validation set
        self.X_val = pd.DataFrame(
            self.scaler.transform(self.X_val),
            columns=self.X_val.columns,
            index=self.X_val.index
        )

        #self.scaler = scaler
        return self


    # preprocess
    def preprocess(self):
        self.load_data().basic_preprocessing().combine_rare_labels().encode_labels().split_features().apply_smote().scale_features()
        return self.X_train.values, self.X_val.values, self.y_train.values, self.y_val.values, self.get_artifacts()









"""
train_data, artifacts = prep.preprocess()

mlflow.log_param("num_features", len(artifacts["features"]))
mlflow.log_param("num_classes", prep.num_classes)
mlflow.log_artifact("scaler.pkl")
mlflow.log_artifact("encoder.pkl")
"""



"""
 para poder pasar un scaler!!!

class MyDataset:
    def __init__(self, pre_cfg, global_cfg, scaler=None):
        self.scaler_type = pre_cfg.scaler_type
        self.scaler = scaler
        if self.scaler is None:
            # instanciar según tipo
            if self.scaler_type == "standard":
                self.scaler = StandardScaler()
            elif self.scaler_type == "minmax":
                self.scaler = MinMaxScaler()
            elif self.scaler_type == "robust":
                self.scaler = RobustScaler()
            else:
                self.scaler = None
"""