import mlflow
from src.core.config import MLFLOW_TRACKING_URI

def init_mlflow():
    """Call this at the start of your entry scripts."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # You can also set default tags here, like the user or environment




# 1. Initialize once at the top
init_mlflow()