"""The project's model: a random forest that predicts the iris species, saved to model.joblib for serving."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/iris.csv")
X, y = df.drop(columns=["species", "petal_size"]), df["species"]
model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
joblib.dump(model, "model.joblib")
print(f"accuracy on the training data: {model.score(X, y):.3f}")
