from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE




class BasePreprocessor(ABC):
    def __init__(self, path, features=None, do_smote=True, smote_strategy=0.3,
                 val_size=0.2, random_state=42):
        self.path = path
        self.features = features
        self.do_smote = do_smote
        self.smote_strategy = smote_strategy
        self.val_size = val_size
        self.random_state = random_state
        
        self.df = None
        self.X_train, self.X_val = None, None
        self.y_train, self.y_val = None, None
        self.label_encoder = None


    # load data and basic preprocessing
    def load_data(self):
        df = pd.read_csv(self.path)
        df = df.drop_duplicates().dropna()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna()
        if self.features:
            df = df[self.features]
        self.df = df
        return self


    def encode_labels(self, rare_attacks=None):
        if rare_attacks is None:
            rare_attacks = ['Bot', 'Web Attack � Brute Force', 'Web Attack � XSS']
        df = self.df.copy()
        df['Label'] = df['Label'].apply(lambda x: 'Other_Attack' if x in rare_attacks else x)
        le = LabelEncoder()
        df['Label'] = le.fit_transform(df['Label'])
        self.df = df
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

    def apply_smote(self):
        if self.do_smote:
            smote = SMOTE(sampling_strategy='auto', random_state=self.random_state) # change later!!!
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

    @abstractmethod
    def preprocess(self):
        pass
