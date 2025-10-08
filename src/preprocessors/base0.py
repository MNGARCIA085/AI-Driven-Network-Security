from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE




class BasePreprocessor(ABC):
    def __init__(self, path, features=None, batch_size=64, balance_factor=0.0, val_size=0.2, random_state=42, scaler_type="standard"):
        self.path = path
        self.features = features
        self.batch_size = batch_size
        self.balance_factor = balance_factor
        self.val_size = val_size
        self.random_state = random_state
        self.scaler_type = scaler_type  # 'standard', 'minmax', 'robust', 'none'
        
        self.df = None
        self.X_train, self.X_val = None, None
        self.y_train, self.y_val = None, None
        self.label_encoder = None
        self.scaler = None


    def load_data(self):
        self.df = pd.read_csv(self.path)
        return self

    def basic_preprocessing(self):
        df = self.df.drop_duplicates().dropna()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna()
        if self.features:
            df = df[self.features]
        self.df = df
        return self


    def combine_rare_labels(self, rare_attacks=None):
        if rare_attacks is None:
            rare_attacks = ['Bot', 'Web Attack � Brute Force', 'Web Attack � XSS']
        df = self.df.copy()
        df['Label'] = df['Label'].apply(lambda x: 'Other_Attack' if x in rare_attacks else x)
        self.df = df
        return self


    def encode_labels(self):
        le = LabelEncoder()
        self.df['Label'] = le.fit_transform(self.df['Label'])
        self.label_encoder = le
        return self


    def split_features(self):
        X = self.df.drop('Label', axis=1)
        y = self.df['Label']
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, stratify=y, random_state=self.random_state
        )
        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val
        return self

    
    def apply_smote(self, balance_factor=0.0):
        """
        balance_factor ∈ [0, 1]:
            0   -> no SMOTE
            1   -> fully balanced (all classes same count)
            <1  -> partially balanced toward the largest class
        """
        if self.balance_factor <= 0:
            return self  # skip SMOTE

        class_counts = self.y_train.value_counts()
        max_count = class_counts.max()

        # build a dict for each class target size
        sampling_strategy = {}
        for cls, count in class_counts.items():
            target = count + (max_count - count) * self.alance_factor
            sampling_strategy[cls] = int(target)

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
        )
        X_res, y_res = smote.fit_resample(self.X_train, self.y_train)
        self.X_train, self.y_train = X_res, y_res
        return self



    @property
    def input_size(self):
        if self.X_train is None:
            raise ValueError("Call split_features() first!")
        return self.X_train.shape[1]

    @property
    def num_classes(self):
        if self.y_train is None:
            raise ValueError("Call split_features() and encode_labels() first!") # after combine rare attacks and encode!!!
        return len(np.unique(self.y_train))

    # for later logging
    def get_artifacts(self):
        """Return objects for logging or serialization."""
        return {
            "scaler": self.scaler,
            "encoder": self.label_encoder,
            "features": self.features,
        }

    @abstractmethod
    def preprocess(self):
        pass
