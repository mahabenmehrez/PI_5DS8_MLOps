"""
Entraine un ElasticNet sur le vin rouge et l'enregistre dans MLflow sous le nom "wine-quality".

  Champion (version 1)  : python tools/train_model.py --alpha 0.01 --alias champion
  Candidat (version 2)  : python tools/train_model.py --alpha 0.5

Les deux modeles ont des predictions differentes, ce qui permet de voir le partage de trafic.
Variables d'environnement : MLFLOW_TRACKING_URI, MLFLOW_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY (comme pour le baseline).
"""
import argparse
import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, default=0.01)
ap.add_argument("--l1", type=float, default=0.5)
ap.add_argument("--alias", default=None, help="alias a poser sur la nouvelle version")
ap.add_argument("--data", default="data/winequality-red.csv")
args = ap.parse_args()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
df = pd.read_csv(args.data, sep=";")
X, y = df.drop(columns=["quality"]), df["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pipe = Pipeline([("scaler", StandardScaler()),
                 ("model", ElasticNet(alpha=args.alpha, l1_ratio=args.l1, random_state=42))])
with mlflow.start_run():
    pipe.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, pipe.predict(X_test))
    mlflow.log_params({"alpha": args.alpha, "l1_ratio": args.l1})
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(pipe, "model", registered_model_name="wine-quality")

client = MlflowClient()
version = str(max(int(v.version) for v in client.search_model_versions("name='wine-quality'")))
if args.alias:
    client.set_registered_model_alias("wine-quality", args.alias, version)
print(f"MAE={mae:.4f}  version={version}  alias={args.alias}")
