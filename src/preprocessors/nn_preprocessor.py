from .base import BasePreprocessor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd




class NNPreprocessor(BasePreprocessor):
    def __init__(self, global_cfg, pre_cfg):
        super().__init__(global_cfg, pre_cfg)


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


    # ---------------------------
    # 2. Test set preprocessing
    # ---------------------------
    def preprocess_test(self, df, scaler, label_encoder, features=None):
        """
        Preprocess a labeled test set using fitted transformers.
        - Cleans and scales features.
        - Combines rare labels (same as training).
        - Encodes labels with the existing label encoder.
        """

        if scaler is None or label_encoder is None: # later -> or self.features is None:
            raise ValueError("Preprocessor missing fitted scaler, encoder, or feature list.")

        # Copy and apply same parent logic
        self.df = df.copy()
        (
            self.basic_preprocessing()
                .combine_rare_labels()  # 🔹 ensure label categories match training
        )

        df = self.df
        
        # cleaner:
        #X = df[self.features]
        #y = df["Label"]
        X = self.df.drop('Label', axis=1)
        y = self.df['Label']

        # Encode labels using the *existing* encoder
        y_encoded = label_encoder.transform(y)

        # Apply scaling (if applicable)
        if self.scaler_type != "none":
            X = pd.DataFrame(scaler.transform(X)) #, columns=self.features)

        return X.values, y_encoded



    # ---------------------------
    # 3. Inference preprocessing (unlabeled dataset)
    # ---------------------------
    def preprocess_inference(self, df, scaler, features=None):
        """Preprocess new unlabeled data (no fitting, no label encoding)."""
        if scaler is None: # or label_encoder is None: # or self.features is None:
            raise ValueError("Preprocessor missing fitted scaler, encoder, or feature list.")

        self.df = df.copy()
        self.basic_preprocessing()

        df = self.df.drop('Label', axis=1)
        # -> for later df = self.df[self.features]; now i already pass it ok (without the label)

        df = pd.DataFrame(scaler.transform(df)) #, columns=self.features)

        return df.values


    # ---------------------------
    # 4. Inference preprocessing (single sample)
    # ---------------------------
    def preprocess_single(self, sample, scaler):
        """Preprocess a single sample for inference."""
        if scaler is None: # or self.features is None:
            raise ValueError("Preprocessor missing fitted scaler or feature list.")

        df = pd.DataFrame([sample]) if not isinstance(sample, pd.DataFrame) else sample
        self.df = df.copy()
        

        #self.basic_preprocessing() .> not needed here, i already pass it ok
        #df = self.df[self.features]
        df = pd.DataFrame(scaler.transform(df)) #, columns=self.features)

        return df.values









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