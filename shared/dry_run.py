"""Dry-run runner (roadmap 5.10, owner P4).

    dry_run(kind, inputs, timeout_s=120) -> DryRunResult(ok, outputs, metrics, log_tail, seconds)

Runs ``dry_run_jobs/<kind>.py`` in a subprocess, inside a fresh temp folder,
with ``DRY_RUN=1`` and ``MLFLOW_EXPERIMENT_NAME=dry_run`` set. It gives
isolation, not logic: each agent writes its own job.

A job module defines one function:

    def run(inputs: dict, workdir: pathlib.Path) -> dict:
        return {"ok": True, "outputs": {...}, "metrics": {...}}

Anything the job prints ends up in ``log_tail``. An exception, a timeout or
a missing ``ok`` key all give ``ok=False``.

Mode: ``subprocess`` (default, used in solo/integration/experiment runs) or
``inprocess`` (unit tests: same contract, no subprocess). Pick it with the
``mode`` argument or the ``DRY_RUN_MODE`` environment variable.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "dry_run_jobs"
LOG_TAIL_LINES = 50
DRY_RUN_EXPERIMENT = "dry_run"


@dataclass
class DryRunResult:
    ok: bool
    outputs: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    log_tail: str = ""
    seconds: float = 0.0


class DryRunViolation(RuntimeError):
    """Raised when code tries to write real state during a dry run."""


def is_dry_run() -> bool:
    return os.environ.get("DRY_RUN") == "1"


def guard_real_write(what: str) -> None:
    """Call before writing a real table, alias or file. shared/tables.py uses this."""
    if is_dry_run():
        raise DryRunViolation(f"refused to write {what!r} while DRY_RUN=1")


def _job_path(kind: str, jobs_dir: Path) -> Path:
    if not kind.replace("_", "").isalnum():
        raise ValueError(f"bad dry-run kind {kind!r}")
    return jobs_dir / f"{kind}.py"


def _tail(text: str, n: int = LOG_TAIL_LINES) -> str:
    return "\n".join(text.splitlines()[-n:])


def _as_result(raw: object, log: str, seconds: float) -> DryRunResult:
    if not isinstance(raw, dict) or "ok" not in raw:
        return DryRunResult(False, log_tail=_tail(log + f"\njob returned {raw!r}, expected a dict with 'ok'"), seconds=seconds)
    return DryRunResult(
        ok=bool(raw["ok"]),
        outputs=dict(raw.get("outputs") or {}),
        metrics=dict(raw.get("metrics") or {}),
        log_tail=_tail(log),
        seconds=seconds,
    )


def dry_run(
    kind: str,
    inputs: dict,
    timeout_s: float = 120,
    *,
    mode: str | None = None,
    jobs_dir: str | Path | None = None,
) -> DryRunResult:
    jobs_dir = Path(jobs_dir) if jobs_dir else JOBS_DIR
    job = _job_path(kind, jobs_dir)
    if not job.exists():
        return DryRunResult(False, log_tail=f"no dry-run job for kind {kind!r} at {job}")
    mode = mode or os.environ.get("DRY_RUN_MODE", "subprocess")
    if mode == "inprocess":
        return _run_inprocess(job, inputs)
    if mode != "subprocess":
        raise ValueError(f"unknown dry-run mode {mode!r}")
    return _run_subprocess(job, inputs, timeout_s)


def _run_subprocess(job: Path, inputs: dict, timeout_s: float) -> DryRunResult:
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"dry_run_{job.stem}_") as tmp:
        workdir = Path(tmp)
        (workdir / "inputs.json").write_text(json.dumps(inputs, default=str))
        env = {
            **os.environ,
            "DRY_RUN": "1",
            "MLFLOW_EXPERIMENT_NAME": DRY_RUN_EXPERIMENT,
            "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])),
        }
        cmd = [sys.executable, "-m", "shared._dry_run_entry", str(job), str(workdir)]
        try:
            proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            parts = (exc.stdout, exc.stderr)
            log = "".join(p.decode(errors="replace") if isinstance(p, bytes) else p for p in parts if p)
            return DryRunResult(False, log_tail=_tail(log + f"\ntimeout after {timeout_s}s"), seconds=time.monotonic() - start)
        log = proc.stdout + proc.stderr
        seconds = time.monotonic() - start
        result_file = workdir / "result.json"
        if proc.returncode != 0 or not result_file.exists():
            return DryRunResult(False, log_tail=_tail(log + f"\nexit code {proc.returncode}"), seconds=seconds)
        return _as_result(json.loads(result_file.read_text()), log, seconds)


@contextlib.contextmanager
def _dry_env():
    saved = {k: os.environ.get(k) for k in ("DRY_RUN", "MLFLOW_EXPERIMENT_NAME")}
    os.environ["DRY_RUN"] = "1"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = DRY_RUN_EXPERIMENT
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_inprocess(job: Path, inputs: dict) -> DryRunResult:
    start = time.monotonic()
    buf = io.StringIO()
    with tempfile.TemporaryDirectory(prefix=f"dry_run_{job.stem}_") as tmp, _dry_env():
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                raw = load_job(job).run(inputs, Path(tmp))
        except Exception:
            return DryRunResult(False, log_tail=_tail(buf.getvalue() + traceback.format_exc()), seconds=time.monotonic() - start)
    return _as_result(raw, buf.getvalue(), time.monotonic() - start)


def load_job(path: Path):
    spec = importlib.util.spec_from_file_location(f"dry_run_jobs.{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise AttributeError(f"{path} has no run(inputs, workdir) function")
    return module
