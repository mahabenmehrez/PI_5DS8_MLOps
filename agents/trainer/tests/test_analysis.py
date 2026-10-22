import numpy as np
import pandas as pd

from agents.trainer import analysis


def test_psi_is_zero_on_reference_and_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cuts = list(np.quantile(ref, np.linspace(0.1, 0.9, 9)))
    share = [0.1] * 10
    assert analysis.psi(rng.normal(0, 1, 5000), cuts, share) < 0.01
    assert analysis.psi(rng.normal(1, 1, 5000), cuts, share) > 0.25


def test_weighted_psi_ignores_unused_features():
    bins = {"features": {"a": {"cuts": [0.0], "ref_share": [0.5, 0.5]}, "b": {"cuts": [0.0], "ref_share": [0.5, 0.5]}}}
    frame = pd.DataFrame({"a": np.r_[-np.ones(50), np.ones(50)], "b": np.ones(100)})  # b moved completely
    out = analysis.drift_table(frame, bins, {"a": 1.0, "b": 0.0})
    assert out["psi"]["b"] > 1 and out["weighted_psi"] < 1e-6


def test_gain_estimate_grows_with_each_signal():
    base = analysis.estimate_gain(0.0, 0.0, 0)
    assert base == 0.0
    assert analysis.estimate_gain(0.1, 0, 0) > base
    assert analysis.estimate_gain(0, 5, 0) > base
    assert analysis.estimate_gain(0, 0, 4000) > base
    assert analysis.estimate_gain(0, -10, 0) == 0.0  # a better live model never adds gain


def test_mae_rise_lower_bound_is_below_point():
    errors = np.random.default_rng(1).exponential(0.55, 1500)
    out = analysis.mae_rise(errors, 0.5)
    assert out["rise_pct_low"] < out["rise_pct"]
    assert abs(out["rise_pct"] - 100 * (errors.mean() / 0.5 - 1)) < 1e-9


def batches(statuses, labelled=None):
    ids = [f"batch-{i:04d}" for i in range(1, len(statuses) + 1)]
    return pd.DataFrame({"batch_id": ids, "status": statuses, "labelled": labelled or [True] * len(ids)})


def test_choose_window_keeps_only_clean_labelled_batches():
    b = batches(["passed", "quarantined", "repaired", "passed", "passed", "passed", "passed"],
                labelled=[True, True, True, True, True, True, False])
    w = analysis.choose_window(b, eval_batches=2, do_not_use={"batch-0004"})
    assert w.eval_ids == ["batch-0005", "batch-0006"]
    assert w.train_ids == ["batch-0001", "batch-0003"]
    assert w.excluded == {"batch-0002": "quarantined", "batch-0004": "do_not_use", "batch-0007": "no_labels_yet"}


def test_choose_window_starts_after_expected_change():
    w = analysis.choose_window(batches(["passed"] * 8), eval_batches=2, after="batch-0003")
    assert w.train_ids == ["batch-0004", "batch-0005", "batch-0006"]
    assert set(w.excluded) == {"batch-0001", "batch-0002", "batch-0003"}


def frame(n=600, seed=0, new_shift=0.0, group_damage=0.0):
    rng = np.random.default_rng(seed)
    q = rng.integers(4, 8, n).astype(float)
    band = np.where(np.arange(n) < n // 3, ">11.5", "<10")
    live = q + rng.normal(0, 0.6, n)
    new = q + rng.normal(0, 0.6, n) * (1 - new_shift) + np.where(band == ">11.5", group_damage, 0.0)
    return pd.DataFrame({"quality": q, "pred_live": live, "pred_new": new, "alcohol_band": band})


def test_gate_passes_a_clearly_better_candidate():
    verdict = analysis.gate(analysis.compare(frame(new_shift=0.3)), max_group_worse_pct=10)
    assert verdict["passed"] and verdict["overall_gain_pct"] > 2


def test_gate_rejects_candidate_worse_for_one_group():
    verdict = analysis.gate(analysis.compare(frame(new_shift=0.5, group_damage=0.8)), max_group_worse_pct=10)
    assert not verdict["passed"]
    assert [(f["group"], f["band"]) for f in verdict["failing_groups"]] == [("alcohol_band", ">11.5")]


def test_gate_rejects_no_gain():
    verdict = analysis.gate(analysis.compare(frame(new_shift=0.0)), max_group_worse_pct=10)
    assert not verdict["overall_ok"] and not verdict["passed"]


def test_small_groups_do_not_count():
    report = analysis.compare(frame(n=120), min_group=50)
    small = report["groups"]["alcohol_band"][">11.5"]
    assert small == {"n": 40, "counts": False}


def test_a_worse_quality_band_is_reported_not_a_veto():
    f = frame(new_shift=0.5, group_damage=0.8).rename(columns={"alcohol_band": "quality_band"})
    verdict = analysis.gate(analysis.compare(f), max_group_worse_pct=10)
    assert verdict["passed"] and verdict["failing_groups"] == []
    assert [(g["group"], g["band"]) for g in verdict["label_groups_worse"]] == [("quality_band", ">11.5")]
