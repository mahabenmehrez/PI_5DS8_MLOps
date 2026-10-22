"""The current model of the project (v2): a random forest on the iris species, saved to model.joblib.
This is the script the pipeline runs; train_old.py is kept only to compare with the first version."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/iris.csv")
X, y = df.drop(columns=["species"]), df["species"]
model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
joblib.dump(model, "model.joblib")
print(f"accuracy on the training data: {model.score(X, y):.3f}")
