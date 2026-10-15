import json, os, random, uuid
from datetime import datetime, timedelta, timezone
import pandas as pd

random.seed(42)
df = pd.read_csv("data/winequality-red.csv", sep=";")
features = [c for c in df.columns if c != "quality"]
start = datetime(2026, 10, 14, 0, 0, tzinfo=timezone.utc)

rows = []
for i in range(400):
    wine = df.iloc[i % len(df)]
    rows.append({
        "request_id": f"req-{uuid.uuid4().hex[:12]}",
        "wine_id": int(i % len(df)),
        "time": (start + timedelta(minutes=3.6 * i)).isoformat(),
        "model_version": "1",
        "variant": "champion",
        "features": json.dumps([float(wine[f]) for f in features]),
        "prediction": float(wine["quality"]) + random.uniform(-0.8, 0.8),
        "latency_ms": round(random.uniform(8, 30), 1),
        "status_code": 200,
    })

os.makedirs("samples", exist_ok=True)
pd.DataFrame(rows).to_parquet("samples/predictions.parquet", index=False)
print("ok:", len(rows), "lignes")