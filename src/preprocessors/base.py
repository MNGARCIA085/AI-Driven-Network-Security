from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class BasePreprocessor(ABC):
    
    def __init__(self, global_cfg, pre_cfg):
        """
        path, features=None, batch_size=64, balance_factor=0.2,
                 val_size=0.2, random_state=42, scaler_type="standard"):
        """

        self.path = pre_cfg.path
        self.features = pre_cfg.features
        self.batch_size = global_cfg.batch_size
        self.balance_factor = pre_cfg.balance_factor
        self.val_size = pre_cfg.val_size
        self.random_state = global_cfg.random_state
        self.scaler_type = pre_cfg.scaler_type  # 'standard', 'minmax', 'robust', 'none'
        self.scaler = None

        # Data placeholders
        self.df = None
        self.X_train = self.X_val = None
        self.y_train = self.y_val = None
        self.label_encoder = None

        # Internal logging
        self._class_dist_before_smote = None
        self._class_dist_after_smote = None


    # ---------------------------
    # 1. Load and clean
    # ---------------------------
    def load_data(self):
        """Load raw data from CSV."""
        self.df = pd.read_csv(self.path)
        return self

    def basic_preprocessing(self):
        """Remove duplicates, NaNs, and infinities."""
        df = self.df.drop_duplicates().dropna()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna()
        if self.features:
            df = df[self.features]
        self.df = df
        return self


    # ---------------------------
    # 2. Label handling
    # ---------------------------
    def combine_rare_labels(self, rare_attacks=None):
        """Group rare classes under a single label."""
        if rare_attacks is None:
            rare_attacks = ['Bot', 'Web Attack � Brute Force', 'Web Attack � XSS']
        df = self.df.copy()
        df['Label'] = df['Label'].apply(lambda x: 'Other_Attack' if x in rare_attacks else x)
        self.df = df
        return self

    def encode_labels(self):
        """Encode target labels to integers."""
        le = LabelEncoder()
        self.df['Label'] = le.fit_transform(self.df['Label'])
        self.label_encoder = le
        return self


    # ---------------------------
    # 3. Split
    # ---------------------------
    def split_features(self):
        """Split dataset into train/validation."""
        X = self.df.drop('Label', axis=1)
        y = self.df['Label']
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, stratify=y, random_state=self.random_state
        )
        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val
        return self


    # ---------------------------
    # 4. SMOTE oversampling
    # ---------------------------
    def apply_smote(self):
        """Apply SMOTE with balance_factor ∈ [0,1]."""
        if self.balance_factor <= 0:
            return self


        #self._class_dist_before_smote = self.y_train.value_counts().to_dict()
        #self._class_dist_after_smote = pd.Series(y_res).value_counts().to_dict()

        class_counts = self.y_train.value_counts()
        self._class_dist_before_smote = class_counts.to_dict()

        max_count = class_counts.max()
        sampling_strategy = {
            cls: int(count + (max_count - count) * self.balance_factor)
            for cls, count in class_counts.items()
        }

        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=self.random_state)
        X_res, y_res = smote.fit_resample(self.X_train, self.y_train)

        self._class_dist_after_smote = pd.Series(y_res).value_counts().to_dict()
        self.X_train, self.y_train = X_res, y_res
        return self


    # ---------------------------
    # 5. Properties
    # ---------------------------
    @property
    def input_size(self):
        if self.X_train is None:
            raise ValueError("Call split_features() first!")
        return self.X_train.shape[1]

    @property
    def num_classes(self):
        if self.y_train is None:
            raise ValueError(
                "Call combine_rare_labels(), encode_labels(), and split_features() first!"
            )
        return len(np.unique(self.y_train))


    # ---------------------------
    # 6. Logging / artifacts
    # ---------------------------
    def get_artifacts(self):
        """Return all key artifacts and metadata for logging."""
        return {
            "features": self.features,
            "scaler_type": self.scaler_type,
            "scaler": self.scaler,
            "encoder": self.label_encoder,
            "input_size": getattr(self, "input_size", None),
            "num_classes": getattr(self, "num_classes", None),
            "balance_factor": self.balance_factor,
            "val_size": self.val_size,
            "random_state": self.random_state,
            "class_dist_before_smote": self._class_dist_before_smote,
            "class_dist_after_smote": self._class_dist_after_smote,
            "train_shape": None if self.X_train is None else self.X_train.shape,
            "val_shape": None if self.X_val is None else self.X_val.shape,
        }


    # ---------------------------
    # 7. Abstract method
    # ---------------------------
    @abstractmethod
    def preprocess(self):
        """Subclasses implement their own full preprocessing pipeline."""
        pass




"""
import mlflow

artifacts = pre.get_artifacts()

with mlflow.start_run():
    # Log numeric and string metadata
    mlflow.log_param("balance_factor", artifacts["balance_factor"])
    mlflow.log_param("val_size", artifacts["val_size"])
    
    # Log class distributions as JSON
    mlflow.log_dict(artifacts["class_dist_before_smote"], "class_dist_before_smote.json")
    mlflow.log_dict(artifacts["class_dist_after_smote"], "class_dist_after_smote.json")
    
    # Optionally log shapes
    mlflow.log_param("train_shape", str(artifacts["train_shape"]))
    mlflow.log_param("val_shape", str(artifacts["val_shape"]))
"""