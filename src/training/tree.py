

from src.models.tree import tree_model


from .base_trainer import BaseTrainer

class TreeTrainer(BaseTrainer):
    def __init__(self, cfg, model_cfg, input_size, num_classes):
        super().__init__(cfg, model_cfg, input_size, num_classes)  # call BaseTrainer init if needed
        self.model = self._create_model()  # instantiate model here


    def _create_model(self):
        # Instantiate your tree or RF model with model_cfg
        return tree_model(self.cfg, self.model_cfg)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_val, y_val):
        y_pred = self.model.predict(X_val)
        return y_pred


    def predict(self, X):
        return self.model.predict(X)


# Random Forest works the same interface as Tree, so you can reuse this class.