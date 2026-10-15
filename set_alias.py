import os
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
client = MlflowClient()
client.set_registered_model_alias("wine-quality", "champion", "1")
print("alias champion -> version 1")