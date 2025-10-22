



class BaseEvaluator:
    def __init__(self, model):
        self.model = model


    def evaluate(self, X, y):
        """Evaluate on labeled data"""
        raise NotImplementedError



    def predict(self, X):
        """see later if i need it"""
        raise NotImplementedError
