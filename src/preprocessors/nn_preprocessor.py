from .base import BasePreprocessor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd




class NNPreprocessor(BasePreprocessor):
    def __init__(self, data_cfg):
        super().__init__(data_cfg)


    def scale_features(self):
        if self.data_cfg.scaler_type == "none":
            return self

        # select scaler type
        scaler_map = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
            "normalize": Normalizer,
        }

        scaler_cls = scaler_map.get(self.data_cfg.scaler_type)
        if scaler_cls is None:
            raise ValueError(f"Unknown scaler type: {self.data_cfg.scaler_type}")
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







    # ---------------------------
    # 2. Test set preprocessing
    # ---------------------------
    def preprocess_test(self, df):
        """
        Preprocess a labeled test set using fitted transformers.
        - Cleans and scales features.
        - Combines rare labels (same as training).
        - Encodes labels with the existing label encoder.
        """

        if self.scaler is None or self.label_encoder is None: # later -> or self.features is None:
            raise ValueError("Preprocessor missing fitted scaler, encoder, or feature list.")

        # Copy and apply same parent logic
        self.df = df.copy()
        (
            self.basic_preprocessing()
                .combine_rare_labels()  # ensure label categories match training
        )

        df = self.df
        
        # cleaner:
        #X = df[self.features]
        #y = df["Label"]
        X = self.df.drop('Label', axis=1)
        y = self.df['Label']

        # Encode labels using the *existing* encoder
        y_encoded = self.label_encoder.transform(y)

        # Apply scaling (if applicable)
        if self.data_cfg.scaler_type != "none":
            X = pd.DataFrame(self.scaler.transform(X)) #, columns=self.features)

        return X.values, y_encoded







    # ---------------------------
    # 3. Inference preprocessing (unlabeled dataset)
    # ---------------------------
    def preprocess_inference(self, df):
        """Preprocess new unlabeled data (no fitting, no label encoding)."""
        if self.scaler is None: # or label_encoder is None: # or self.features is None:
            raise ValueError("Preprocessor missing fitted scaler, encoder, or feature list.")

        self.df = df.copy()

        self.basic_preprocessing()

        df = self.df.drop('Label', axis=1) if 'Label' in self.df.columns else self.df
        # -> for later df = self.df[self.data_cfg.features]; now i already pass it ok (without the label)

        df = pd.DataFrame(self.scaler.transform(df)) #, columns=self.data_cfg.features)

        return df.values


    # ---------------------------
    # 4. Inference preprocessing (single sample)
    # ---------------------------
    def preprocess_single(self, sample):
        """Preprocess a single sample for inference."""
        if self.scaler is None: # or self.features is None:
            raise ValueError("Preprocessor missing fitted scaler or feature list.")

        df = pd.DataFrame([sample]) if not isinstance(sample, pd.DataFrame) else sample
        self.df = df.copy()

        #self.basic_preprocessing() .> not needed here, i already pass it ok
        #df = self.df[self.features]
        df = pd.DataFrame(self.scaler.transform(df)) #, columns=self.features)

        return df.values


    











"""


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