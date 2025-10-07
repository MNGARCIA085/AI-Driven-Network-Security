from .base import BasePreprocessor

class TreePreprocessor(BasePreprocessor):
    def preprocess(self):
        self.load_data().encode_labels().split_features().apply_smote()
        return self.X_train.values, self.X_val.values, self.y_train.values, self.y_val.values
