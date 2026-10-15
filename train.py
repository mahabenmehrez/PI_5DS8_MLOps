import os
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

df = pd.read_csv("data/winequality-red.csv", sep=";")
X, y = df.drop(columns=["quality"]), df["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pipe = Pipeline([("scaler", StandardScaler()),
                 ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42))])

with mlflow.start_run():
    pipe.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, pipe.predict(X_test))
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(pipe, "model", registered_model_name="wine-quality")
    print("MAE:", mae)

client = MlflowClient()
last = max(int(v.version) for v in client.search_model_versions("name='wine-quality'"))
client.set_registered_model_alias("wine-quality", "champion", last)
print("alias champion -> version", last)