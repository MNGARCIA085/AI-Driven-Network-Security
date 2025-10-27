


class BaseEvaluator:
    def __init__(self, cfg, model):
        # cfg -> eval config
        self.model = model
        self.average = cfg.metrics.average

    
    def evaluate(self, X, y):
        """Evaluate on labeled data"""
        raise NotImplementedError


    def predict(self, X):
        """see later if i need it"""
        raise NotImplementedError
