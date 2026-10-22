"""
Fault runner (owner P3, roadmap 13.4). Loads every harness/faults/<layer>/*.yaml,
applies the active faults to the replayed feed, and writes the ground truth to
the `faults` table, which agents never read.

    python -m harness.fault_runner list
"""
from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import yaml

FAULT_DIR = Path(__file__).resolve().parent / "faults"


def load_callable(ref: str) -> Callable:
    module, func = ref.split(":")
    return getattr(importlib.import_module(module), func)


@dataclass
class Fault:
    id: str
    layer: str
    target: dict
    after_batches: int
    duration: int
    applies_to: list[str]
    inject: str
    revert: str
    expected: dict = field(default_factory=dict)
    author: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> "Fault":
        d = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(id=d["id"], layer=d["layer"], target=d.get("target", {}),
                   after_batches=int(d["start"]["after_batches"]),
                   duration=int(d.get("duration", {}).get("batches", 1)),
                   applies_to=list(d.get("applies_to", ["batch"])), inject=d["inject"],
                   revert=d.get("revert", ""), expected=d.get("expected", {}),
                   author=d.get("author", ""))

    def active(self, batch_index: int) -> bool:
        """batch_index is 1-based: after_batches=10 hits batches 11 .. 10+duration."""
        return self.after_batches < batch_index <= self.after_batches + self.duration

    def apply(self, df: pd.DataFrame, batch_index: int) -> pd.DataFrame:
        """Two inject styles: ours f(df, **target), and the Trainer's f(batch, step, target)
        (harness/faults/model.py) where step counts the batches inside the fault from 0."""
        inject = load_callable(self.inject)
        if "step" in inspect.signature(inject).parameters:
            return inject(df, batch_index - self.after_batches - 1, dict(self.target))
        return inject(df, **self.target)


def all_faults() -> dict[str, Fault]:
    faults = {}
    for path in sorted(FAULT_DIR.glob("*/*.yaml")):
        f = Fault.from_yaml(path)
        faults[f.id] = f
    return faults


class FaultRunner:
    def __init__(self, faults: list[Fault], tables=None, run_id: str = "dev"):
        self.faults = faults
        self.tables = tables
        self.run_id = run_id
        self.recorded: set[str] = set()

    def apply(self, batch_index: int, df: pd.DataFrame, where: str,
              batch_id: Optional[str] = None) -> pd.DataFrame:
        for f in self.faults:
            if f.active(batch_index) and where in f.applies_to:
                df = f.apply(df, batch_index)
                if self.tables is not None and f.id not in self.recorded:
                    self.recorded.add(f.id)
                    self.tables.record_fault(f.id, f.layer, f.target, batch_id or str(batch_index),
                                             f.expected, self.run_id)
        return df


if __name__ == "__main__":
    if sys.argv[1:] == ["list"]:
        for fid, f in all_faults().items():
            print(f"{fid:45s} layer={f.layer:6s} batches {f.after_batches + 1}-{f.after_batches + f.duration}"
                  f"  expected={f.expected.get('acceptable_actions')}")
