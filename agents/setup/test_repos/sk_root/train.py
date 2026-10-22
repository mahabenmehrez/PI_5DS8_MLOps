"""Train a logistic regression on the iris data and save it to model.joblib."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/iris.csv")
X, y = df.drop(columns=["species"]), df["species"]
model = LogisticRegression(max_iter=500).fit(X, y)
joblib.dump(model, "model.joblib")
print(f"accuracy on the training data: {model.score(X, y):.3f}")
