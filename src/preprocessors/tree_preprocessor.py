from .base import BasePreprocessor
import pandas as pd


class TreePreprocessor(BasePreprocessor):
    
    def __init__(self, global_cfg, pre_cfg):
        super().__init__(global_cfg, pre_cfg)
        # Trees don't need scaling
        # self.scaler = None

    def preprocess(self):
        self.load_data().basic_preprocessing().combine_rare_labels().encode_labels().split_features().apply_smote()
        return self.X_train.values, self.X_val.values, self.y_train.values, self.y_val.values, self.get_artifacts()


    #########TO CHECK######################
    def preprocess_test(self, df, label_encoder, features=None): # for label test data
        """
        Preprocess a labeled test set for tree-based models.
        - Cleans and aligns features.
        - Combines rare labels.
        - Encodes labels with existing encoder (no scaling).
        """
        if label_encoder is None:
            raise ValueError("Preprocessor missing fitted label encoder.")

        self.df = df.copy()
        (
            self.basic_preprocessing()
                .combine_rare_labels()
        )

        X = self.df.drop('Label', axis=1)
        y = self.df['Label']

        # Encode target labels
        y_encoded = label_encoder.transform(y)

        # No scaling for tree models
        return X.values, y_encoded


    def preprocess_inference(self, df, features=None):
        """
        Preprocess new unlabeled data for tree-based models (no scaling, no label encoding).
        """
        self.df = df.copy()
        self.basic_preprocessing()

        X = self.df.drop('Label', axis=1) if 'Label' in self.df.columns else self.df
        return X.values


    def preprocess_single(self, sample):
        """
        Preprocess a single sample for tree-based models (no scaling).
        """
        df = pd.DataFrame([sample]) if not isinstance(sample, pd.DataFrame) else sample
        self.df = df.copy()
        # No scaling, no transformations needed
        return df.values



