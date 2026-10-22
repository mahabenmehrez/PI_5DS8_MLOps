"""
Sample Pandera schema WineBatch (P1 + P3, roadmap 6.4). The Setup Agent will
generate the real one in week 2; consumers build against this sample.

Ranges are physical limits that hold for red AND white wines (UCI wine quality
data), not the red-wine percentiles: a valid white-wine batch must pass the
schema and be judged by the drift checks instead. Units follow agents.yaml.
"""
try:
    import pandera.pandas as pa
except ImportError:            # pandera < 0.24
    import pandera as pa

FEATURES = {
    # column name in the CSV: (min, max)   unit
    "fixed acidity": (3.0, 16.0),           # g/dm3 tartaric acid
    "volatile acidity": (0.05, 2.0),        # g/dm3 acetic acid
    "citric acid": (0.0, 2.0),              # g/dm3
    "residual sugar": (0.0, 70.0),          # g/dm3
    "chlorides": (0.005, 0.7),              # g/dm3 sodium chloride
    "free sulfur dioxide": (1.0, 300.0),    # mg/dm3
    "total sulfur dioxide": (1.0, 450.0),   # mg/dm3 (g/dm3 values fall below 1)
    "density": (0.98, 1.01),                # g/cm3
    "pH": (2.5, 4.5),
    "sulphates": (0.2, 2.5),                # g/dm3 potassium sulphate
    "alcohol": (7.0, 16.0),                 # % vol
}

def make_schema(features: dict = FEATURES) -> "pa.DataFrameSchema":
    """WineBatch for a given {column: (min, max)}; a new version after an approved unit change."""
    return pa.DataFrameSchema(
        {name: pa.Column(float, pa.Check.in_range(lo, hi), nullable=False, coerce=True)
         for name, (lo, hi) in features.items()}
        | {"quality": pa.Column(int, pa.Check.in_range(0, 10), nullable=True, required=False, coerce=True),
           "wine_id": pa.Column(int, required=False, coerce=True)},
        strict=False,          # new columns are reported by the Data Checker, not rejected here
    )


WineBatch = make_schema()
