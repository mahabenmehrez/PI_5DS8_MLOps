import pytest

from baseline.app.release import ReleaseManager


class FakeModel:
    def predict(self, df):
        return [5.0]


def make(tmp_path):
    loaded = []

    def loader(uri):
        loaded.append(uri)
        return FakeModel()

    return ReleaseManager(db_path=str(tmp_path / "b.db"), loader=loader), loaded


def set_state(rm, **fields):
    v = {"champion_version": "1", "candidate_version": None, "stage": "none",
         "candidate_share": 0.0, "shadow_on": 0, **fields}
    con = rm._connect()
    con.execute(
        "INSERT OR REPLACE INTO release_state VALUES (1,?,?,?,?,?,?)",
        (v["champion_version"], v["candidate_version"], v["stage"],
         v["candidate_share"], v["shadow_on"], "2026-01-01T00:00:00+00:00"))
    con.commit()
    con.close()


def test_starts_with_champion_only(tmp_path):
    rm, loaded = make(tmp_path)
    rm.refresh()
    assert loaded == ["models:/wine-quality@champion"]
    assert rm.route() == ("champion", False)


def test_candidate_loaded_and_shadow_mode(tmp_path):
    rm, loaded = make(tmp_path)
    set_state(rm, candidate_version="2", stage="shadow", shadow_on=1)
    rm.refresh()
    assert "models:/wine-quality/2" in loaded
    assert rm.route() == ("champion", True)


def test_ten_percent_split(tmp_path):
    rm, _ = make(tmp_path)
    set_state(rm, candidate_version="2", stage="10%", candidate_share=0.10)
    rm.refresh()
    n = sum(rm.route()[0] == "candidate" for _ in range(5000))
    assert 350 < n < 650


def test_candidate_dropped_after_rollback(tmp_path):
    rm, _ = make(tmp_path)
    set_state(rm, candidate_version="2", stage="10%", candidate_share=0.10)
    rm.refresh()
    set_state(rm, candidate_version=None, stage="none")
    rm.refresh()
    assert rm.candidate is None and rm.route() == ("champion", False)


def test_admin_apply_is_immediate(tmp_path):
    rm, _ = make(tmp_path)
    set_state(rm, candidate_version="2", stage="10%", candidate_share=0.10)
    rm.refresh()
    rm.apply(candidate_share=0.0)
    assert all(rm.route()[0] == "champion" for _ in range(200))


def test_log_flush_and_agent_can_read_it(tmp_path, monkeypatch):
    rm, _ = make(tmp_path)
    rm.log("champion", "1", 7, [1, 2], 5.1, 20.0, 200)
    rm.log("candidate", "2", None, [1], None, 30.0, 500)
    assert rm.flush() == 2
    monkeypatch.setenv("BOARD_DB", rm.db_path)
    from agents.deployer import state
    rows = state.read_predictions("candidate")
    assert len(rows) == 1 and rows[0]["status_code"] == 500


def test_candidate_fault_injection_from_env(tmp_path, monkeypatch):
    rm, _ = make(tmp_path)
    monkeypatch.setenv("CANDIDATE_ERR_PCT", "100")
    with pytest.raises(RuntimeError):
        rm.candidate_fault()


def test_runtime_fault_overrides_env(tmp_path, monkeypatch):
    rm, _ = make(tmp_path)
    monkeypatch.setenv("CANDIDATE_ERR_PCT", "100")
    rm.set_fault(slow_ms=0, err_pct=0)                 # la faute a chaud gagne sur l'env
    rm.candidate_fault()
    rm.set_fault(err_pct=100)
    with pytest.raises(RuntimeError):
        rm.candidate_fault()


def test_forced_candidate_for_replay(tmp_path):
    rm, _ = make(tmp_path)
    set_state(rm, candidate_version="2", stage="10%", candidate_share=0.0)
    rm.refresh()
    assert rm.route() == ("champion", False)           # part a 0 %
    assert rm.route("candidate") == ("candidate", False)
    set_state(rm, candidate_version=None, stage="none")
    rm.refresh()
    assert rm.route("candidate") == ("champion", False)   # pas de candidat charge


def test_fault_plan_returns_delay_without_sleeping(tmp_path):
    rm, _ = make(tmp_path)
    rm.set_fault(slow_ms=400, err_pct=0)
    start = __import__("time").time()
    assert rm.candidate_fault_plan() == 0.4
    assert __import__("time").time() - start < 0.1          # le delai est applique par l'API, en async


def test_fault_plan_raises_injected_error(tmp_path):
    rm, _ = make(tmp_path)
    rm.set_fault(slow_ms=0, err_pct=100)
    with pytest.raises(RuntimeError):
        rm.candidate_fault_plan()
