from .base import BasePreprocessor

class TreePreprocessor(BasePreprocessor):
    
    def __init__(self, global_cfg, pre_cfg):
        super().__init__(global_cfg, pre_cfg)
        # Trees don't need scaling
        # self.scaler = None

    def preprocess(self):
        self.load_data().basic_preprocessing().combine_rare_labels().encode_labels().split_features().apply_smote()
        return self.X_train.values, self.X_val.values, self.y_train.values, self.y_val.values,self.get_artifacts()
