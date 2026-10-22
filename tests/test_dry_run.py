import os
from pathlib import Path

import pytest

from shared.dry_run import DryRunViolation, dry_run, guard_real_write

JOBS = Path(__file__).parent / "jobs"
ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("mode", ["subprocess", "inprocess"])
def test_ok_job(mode):
    res = dry_run("echo", {"x": 21}, mode=mode, jobs_dir=JOBS)
    assert res.ok and res.metrics == {"x": 42}
    assert res.outputs["dry_run"] == "1"
    assert "echo job running" in res.log_tail
    assert os.environ.get("DRY_RUN") != "1"  # the caller's env is untouched


@pytest.mark.parametrize("mode", ["subprocess", "inprocess"])
def test_failing_job(mode):
    res = dry_run("crash", {}, mode=mode, jobs_dir=JOBS)
    assert not res.ok and "boom" in res.log_tail


def test_job_reporting_not_ok():
    assert not dry_run("echo", {"ok": False}, jobs_dir=JOBS).ok


def test_timeout():
    res = dry_run("slow", {"sleep": 5}, timeout_s=1, jobs_dir=JOBS)
    assert not res.ok and "timeout" in res.log_tail


def test_missing_job():
    assert not dry_run("nope", {}, jobs_dir=JOBS).ok


def test_guard(monkeypatch):
    guard_real_write("batches")  # fine outside a dry run
    monkeypatch.setenv("DRY_RUN", "1")
    with pytest.raises(DryRunViolation):
        guard_real_write("batches")


def test_train_in_dry_run_registers_nothing():
    res = dry_run("train_smoke", {"data": str(ROOT / "data" / "winequality-red.csv")}, timeout_s=180, jobs_dir=JOBS)
    assert res.ok, res.log_tail
    assert res.outputs["version"] is None and res.metrics["mae"] < 0.6
