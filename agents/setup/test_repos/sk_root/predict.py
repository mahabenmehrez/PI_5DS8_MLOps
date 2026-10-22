"""Predict the species of one flower: python predict.py 5.1 3.5 1.4 0.2"""
import sys

import joblib
import pandas as pd

COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

if __name__ == "__main__":
    model = joblib.load("model.joblib")
    row = pd.DataFrame([[float(v) for v in sys.argv[1:5]]], columns=COLUMNS)
    print(int(model.predict(row)[0]))
