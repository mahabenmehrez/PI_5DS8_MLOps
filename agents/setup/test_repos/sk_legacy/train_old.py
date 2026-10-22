"""First version of the model (v1, replaced by train_v2.py): a logistic regression, saved to model_v1.joblib.
Kept only to compare the two versions; the pipeline doesn't run it."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/iris.csv")
X, y = df.drop(columns=["species"]), df["species"]
model = LogisticRegression(max_iter=500).fit(X, y)
joblib.dump(model, "model_v1.joblib")
print(f"accuracy on the training data: {model.score(X, y):.3f}")
