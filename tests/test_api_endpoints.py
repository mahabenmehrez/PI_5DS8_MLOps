"""Teste main.py avec un vrai client FastAPI et de faux modeles (mlflow est remplace)."""
import sys
import time
import types

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


class FakeModel:
    def __init__(self, value):
        self.value = value

    def predict(self, df):
        if df.shape[1] != 11 or len(df.columns) != 11 or df.isna().any().any() and False:
            raise ValueError("mauvais nombre de colonnes")
        return [self.value]


@pytest.fixture
def api(tmp_path, monkeypatch):
    fake = types.ModuleType("mlflow")
    fake.set_tracking_uri = lambda uri: None
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv("BOARD_DB", str(tmp_path / "board.db"))
    monkeypatch.delitem(sys.modules, "baseline.app.main", raising=False)
    from fastapi.testclient import TestClient
    from baseline.app import main

    def loader(uri):
        return FakeModel(7.0 if uri.endswith("/2") else 5.0)

    main.release.loader = loader
    main.release.db_path = str(tmp_path / "board.db")
    with TestClient(main.app) as client:
        yield client, main


BODY = {"data": [7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4], "wine_id": 3}


def set_candidate(main, share=0.0, stage="10%", shadow=0):
    con = main.release._connect()
    con.execute("INSERT OR REPLACE INTO release_state VALUES (1,'1','2',?,?,?,?)",
                (stage, share, shadow, "2026-01-01T00:00:00+00:00"))
    con.commit()
    con.close()
    main.release.refresh()


def test_predict_uses_champion_and_logs(api):
    client, main = api
    r = client.post("/predict", json=BODY)
    assert r.status_code == 200 and r.json() == {"prediction": 5.0}
    assert r.headers["X-Served-Variant"] == "champion"
    main.release.flush()
    from agents.deployer import state
    rows = state.read_predictions("champion")
    assert len(rows) == 1 and rows[0]["wine_id"] == 3 and rows[0]["prediction"] == 5.0


def test_admin_release_moves_traffic_immediately(api):
    client, main = api
    set_candidate(main, share=0.0)
    assert client.post("/predict", json=BODY).json() == {"prediction": 5.0}
    client.post("/admin/release", json={"candidate_share": 1.0})
    assert client.post("/predict", json=BODY).json() == {"prediction": 7.0}
    assert client.get("/admin/release").json()["candidate_loaded"] == "2"


def test_dry_run_forces_candidate_and_writes_nothing(api):
    client, main = api
    set_candidate(main, share=0.0)
    r = client.post("/predict", json=BODY, headers={"X-Variant": "candidate", "X-Dry-Run": "1"})
    assert r.json() == {"prediction": 7.0} and r.headers["X-Served-Variant"] == "candidate"
    main.release.flush()
    from agents.deployer import state
    assert state.read_predictions() == []


def test_shadow_logs_candidate_but_champion_answers(api):
    client, main = api
    set_candidate(main, stage="shadow", shadow=1)
    assert client.post("/predict", json=BODY).json() == {"prediction": 5.0}
    main.release.flush()
    from agents.deployer import state
    assert len(state.read_predictions("shadow")) == 1
    assert state.read_predictions("shadow")[0]["prediction"] == 7.0


def test_runtime_fault_makes_candidate_fail_and_logs_500(api):
    client, main = api
    set_candidate(main, share=1.0)
    client.post("/admin/fault", json={"err_pct": 100})
    with pytest.raises(Exception):
        client.post("/predict", json=BODY)
    main.release.flush()
    from agents.deployer import state
    assert state.read_predictions("candidate")[0]["status_code"] == 500
    client.post("/admin/fault", json={})                     # supprime les fautes
    assert client.post("/predict", json=BODY).json() == {"prediction": 7.0}


def test_runtime_fault_slows_candidate(api):
    client, main = api
    set_candidate(main, share=1.0)
    client.post("/admin/fault", json={"slow_ms": 150})
    start = time.time()
    client.post("/predict", json=BODY)
    assert time.time() - start >= 0.15


def test_scenario_h_bug_is_kept(api):
    client, _ = api
    with pytest.raises(UnboundLocalError):
        client.post("/predict", json={"data": [1, 2]})
