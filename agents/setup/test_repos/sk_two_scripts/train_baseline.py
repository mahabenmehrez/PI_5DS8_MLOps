"""Side experiment, not the project's model: can the sepal measures alone tell small petals from large ones?
Prints a score and saves nothing."""
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/iris.csv")
X, y = df[["sepal_length", "sepal_width"]], df["petal_size"]
clf = LogisticRegression(max_iter=500).fit(X, y)
print(f"petal size from sepals: {clf.score(X, y):.3f}")
