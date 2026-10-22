"""Train a random forest on the iris data (run as a module: python -m irisclf.train)."""
import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

from .features import add_ratios

CONFIG = {"n_estimators": 50, "random_state": 0}

if __name__ == "__main__":
    df = add_ratios(pd.read_csv("data/iris.csv"))
    X, y = df.drop(columns=["species"]), df["species"]
    model = RandomForestClassifier(**CONFIG).fit(X, y)
    Path("models").mkdir(exist_ok=True)
    Path("models/forest.pkl").write_bytes(pickle.dumps(model))
    Path("models/config.yaml").write_text(yaml.safe_dump(CONFIG))
    print(f"accuracy on the training data: {model.score(X, y):.3f}")
