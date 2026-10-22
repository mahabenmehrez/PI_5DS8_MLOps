"""
Dry-run job `repair_check` (written by the Data Checker, roadmap 5.10).

batch, repair + params -> repaired copy, checks passed, PSI after.
Runs on a copy in a temp folder; with DRY_RUN=1 the shared tables refuse writes.

    inputs = {"action": "repair",
              "params": {"batch_id": ..., "path": <batch file>, "steps": [...],
                         "profile_path": <profile_vN.json written by the Checker>}}

Run by shared/dry_run.py (subprocess or in-process): run(inputs, workdir) -> {ok, outputs,
metrics}; whatever it prints ends up in the result's log_tail.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from agents.checker.analysis import repair_until_clean
from agents.checker.checks import Inspector, column_psi
from samples.schema import make_schema

PSI_LIMIT = 0.25


def run(inputs: dict, workdir: Optional[Path] = None) -> dict:
    params = inputs["params"]
    saved = json.loads(Path(params["profile_path"]).read_text(encoding="utf-8"))
    profile, cfg = saved["profile"], saved["cfg"]
    bounds = {k: tuple(v) for k, v in saved["bounds"].items()}
    inspector = Inspector(profile, cfg, make_schema(bounds))
    min_score = cfg["thresholds"].get("decision_min_score", 0.5)

    df = pd.read_parquet(params["path"])
    repaired, applied, remaining = repair_until_clean(df, params["steps"], inspector, profile,
                                                      bounds, min_score)
    touched = sorted({s["params"].get("column") or s["params"].get("old")
                      for s in applied} - {None})
    # PSI above the sampling noise of a small batch (same correction as the checks)
    psi_after = {}
    for c in touched:
        if c in repaired.columns and c in profile["columns"]:
            ref = profile["columns"][c]
            noise = (len(ref["shares"]) - 1) * (1 / max(len(repaired), 1) + 1 / profile["rows"])
            psi_after[c] = round(column_psi(repaired[c], ref) - noise, 4)
    folder = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="dry_run_"))
    out_path = folder / f"{params['batch_id']}.parquet"
    repaired.to_parquet(out_path, index=False)

    ok = not remaining and all(v < PSI_LIMIT for v in psi_after.values())
    if not ok:
        print(f"repair_check failed: still {[f.check for f in remaining]}, psi_after={psi_after}")
    return {"ok": ok,
            "outputs": {"repaired_path": str(out_path), "steps": applied,
                        "remaining": [f.to_dict() for f in remaining]},
            "metrics": {"rows_before": len(df), "rows_after": len(repaired), "psi_after": psi_after}}
