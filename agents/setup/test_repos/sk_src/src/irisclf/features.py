import pandas as pd


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["petal_ratio"] = out["petal_length"] / out["petal_width"]
    out["sepal_ratio"] = out["sepal_length"] / out["sepal_width"]
    return out
