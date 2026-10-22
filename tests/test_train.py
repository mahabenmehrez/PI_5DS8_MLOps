import json

import mlflow
import pandas as pd
import pytest

from baseline import train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("train")
    df = train.load_wines(train.DEFAULT_DATA)
    res = train.train(df, tracking_uri=f"file:{tmp}/mlruns", reference_dir=tmp / "ref")
    return res, tmp


def test_data_shape():
    df = train.load_wines(train.DEFAULT_DATA)
    assert df.shape == (1599, 13) and df["wine_id"].is_unique
    white = train.load_wines(train.DEFAULT_DATA.with_name("winequality-white.csv"))
    assert white.shape == (4898, 13)


def test_prepare_features_accepts_both_namings():
    df = train.load_wines(train.DEFAULT_DATA).head(5)
    X1, y1 = train.prepare_features_train(df)
    X2, _ = train.prepare_features_train(df.rename(columns=lambda c: c.replace(" ", "_")))
    assert list(X1.columns) == train.RAW_FEATURES and X1.equals(X2) and len(y1) == 5
    with pytest.raises(ValueError):
        train.prepare_features_train(df.drop(columns=["alcohol"]))


def test_registered_as_champion(trained):
    res, tmp = trained
    assert res.version == "1" and res.alias == "champion"
    assert res.train_rows == 1199 and res.holdout_rows == 400
    assert 0.45 < res.mae < 0.55
    client = mlflow.MlflowClient()
    assert str(client.get_model_version_by_alias("wine-quality", "champion").version) == "1"
    model = mlflow.pyfunc.load_model("models:/wine-quality@champion")
    row = pd.DataFrame([[7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]], columns=train.RAW_FEATURES)
    assert 3 < model.predict(row)[0] < 8  # integers in the input are fine, like app/main.py sends


def test_reference_files(trained):
    res, tmp = trained
    ref = tmp / "ref"
    frame = pd.read_parquet(ref / "reference.parquet")
    assert len(frame) == 400
    assert {"wine_id", *train.FEATURES, "quality", "prediction", "alcohol_band", "acidity_band", "quality_band"} <= set(frame)
    imp = json.loads((ref / "feature_importance.json").read_text())
    assert set(imp) == set(train.FEATURES) and max(imp, key=imp.get) == "alcohol"
    bins = json.loads((ref / "psi_bins.json").read_text())
    for f in train.FEATURES:
        b = bins["features"][f]
        assert len(b["ref_share"]) == len(b["cuts"]) + 1 and abs(sum(b["ref_share"]) - 1) < 1e-9
    gm = json.loads((ref / "group_metrics.json").read_text())
    assert gm["overall"]["n"] == 400 and abs(gm["overall"]["mae"] - res.mae) < 1e-9
    assert set(gm["groups"]) == {"alcohol_band", "acidity_band", "quality_band"}
    # the reference files are also logged with the run
    files = {a.path for a in mlflow.MlflowClient().list_artifacts(res.run_id, "reference")}
    assert {"reference/reference.parquet", "reference/psi_bins.json"} <= files


def test_groups():
    df = pd.DataFrame({"alcohol": [9.9, 10.0, 11.5, 11.6], "fixed acidity": [7.4, 7.5, 9.5, 9.6], "quality": [5, 6, 7, 3]})
    g = train.assign_groups(df)
    assert list(g["alcohol_band"]) == ["<10", "10-11.5", "10-11.5", ">11.5"]
    assert list(g["acidity_band"]) == ["<7.5", "7.5-9.5", "7.5-9.5", ">9.5"]
    assert list(g["quality_band"]) == ["<=5", "6", ">=7", "<=5"]
