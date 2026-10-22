"""The errors the demo can inject. Each one is a small piece of code added to the
training script of a copy of the project (never the real one), like the evaluation's faults.

The code goes after the module docstring and the `from __future__` imports, so it works with any train.py.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Fault:
    id: str
    label: str            # the button text
    what: str             # what the error is, in one sentence
    code: str             # added to the training script
    expected: str         # what a good agent does, said before the run
    needs_llm: bool       # True: no rule knows it; the rules alone report it
    expect: str = "build"  # the right outcome, as in the evaluation: "build" (fixed) or "report" (setup_failed)


FAULTS = [
    Fault("hidden_dependency", "A package loaded in secret",
          "The code loads the emoji package by name at run time: scanning the code cannot see it, and it is "
          "declared nowhere.",
          'import importlib\nimportlib.import_module("emoji")\n',
          "A rule is enough: the ModuleNotFoundError names the package, the agent adds it and runs again.",
          needs_llm=False),
    Fault("numpy_float", "A function removed from numpy",
          "np.float was removed in numpy 1.24: the code was written for an older numpy.",
          "import numpy as _np\n_np.float(1.0)\n",
          "No rule knows this error. The LLM must find a numpy version that still has np.float "
          "(before 1.24) and works with the rest of the project; the dry run checks it.",
          needs_llm=True),
    Fault("sklearn_load_boston", "A function removed from scikit-learn",
          "load_boston was removed in scikit-learn 1.2: the code was written for an older version.",
          "from sklearn.datasets import load_boston as _lb\n",
          "No rule knows this error. The LLM must choose a scikit-learn version before 1.2 "
          "(for example 1.1.3), sometimes numpy too; every attempt is checked by a dry run.",
          needs_llm=True),
    Fault("team_typo", "A typo in the team's code",
          "The code imports ElasticNett (with two t's), a name that exists in no version of scikit-learn.",
          "from sklearn.linear_model import ElasticNett\n",
          "Fixing the team's code is not the agent's job: no version fixes this. The right behaviour is to "
          "give up and report the error (setup_failed) to its owner.",
          needs_llm=False, expect="report"),
    Fault("missing_data_file", "A data file that does not exist",
          "The code reads data/winequality-rose.csv, a file that exists nowhere in the project.",
          'open("data/winequality-rose.csv").close()\n',
          "No installation creates a data file: the agent must report it, without installing anything.",
          needs_llm=False, expect="report"),
]
BY_ID = {f.id: f for f in FAULTS}


def insert_at(source: str) -> int:
    """The line index after the module docstring and the `from __future__` imports (where code can go)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    at = 0
    for i, node in enumerate(tree.body):
        doc = i == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) \
            and isinstance(node.value.value, str)
        future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        if not (doc or future):
            break
        at = node.end_lineno
    return at


def inject(script: Path, fault: Fault, shown: str | None = None) -> str:
    """Add the fault's code to the script; returns where it went ("baseline/train.py, line 17"), for the page."""
    text = script.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    at = insert_at(text)
    script.write_text("".join(lines[:at]) + fault.code + "".join(lines[at:]), encoding="utf-8")
    return f"{shown or script.name}, line {at + 1}"
