import numpy as np
from agents.deployer import analysis as a

OK = {"ok": True, "waiting": [], "bad": []}


def win(n=100, p95=25.0, err=0.0):
    return {"n": n, "p95_ms": p95, "error_rate": err}


def test_hard_limit_needs_two_minutes():
    assert a.hard_limit_breached([win(err=0.08)]) is None
    assert a.hard_limit_breached([win(err=0.08), win(err=0.09)])["metric"] == "error_rate"
    assert a.hard_limit_breached([win(p95=420), win(p95=410)])["metric"] == "p95_ms"
    assert a.hard_limit_breached([win(err=0.08), win(err=0.0)]) is None


def test_errors_not_worse():
    rng = np.random.default_rng(0)
    champ = np.abs(rng.normal(0.5, 0.2, 200))
    same = np.abs(rng.normal(0.5, 0.2, 200))
    worse = np.abs(rng.normal(1.0, 0.2, 200))
    assert a.errors_not_worse(same, champ)
    assert not a.errors_not_worse(worse, champ)


def test_gate_waits_then_passes_at_10():
    champ, rng = win(p95=25), np.random.default_rng(1)
    errs = list(np.abs(rng.normal(0.5, 0.2, 150)))
    g = a.evaluate_gate("10%", win(n=50), champ, errs, errs, minutes_in_stage=10)
    assert g["waiting"] and not g["ok"]
    g = a.evaluate_gate("10%", win(n=200, p95=27), champ, errs, errs, minutes_in_stage=40)
    assert g["ok"]


def test_gate_flags_slow_candidate():
    errs = list(np.abs(np.random.default_rng(2).normal(0.5, 0.2, 150)))
    g = a.evaluate_gate("10%", win(n=200, p95=80), win(p95=25), errs, errs, 40)
    assert g["bad"] and not g["ok"]


def test_rule_decisions():
    assert a.rule_decision("10%", OK, {"metric": "p95_ms", "value": 400, "limit": 300}, [])["action"] == "rollback_emergency"
    assert a.rule_decision("10%", OK, None, ["unit change"])["action"] == "hold"
    d = a.rule_decision("10%", OK, None, [])
    assert d["action"] == "advance_stage" and d["params"]["to_stage"] == "50%"
    d = a.rule_decision("50%", OK, None, [])
    assert d["action"] == "promote_100" and d["level"] == 3
    bad = {"ok": False, "waiting": [], "bad": ["slow"]}
    assert a.rule_decision("10%", bad, None, [])["action"] == "hold"