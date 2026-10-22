"""Team mode: complete the team's wine project with the files the other agents need (roadmap 7.1).

    from builder_agent.team import TeamLayout, team_plan, render_team, write_team
    plan = team_plan(TeamLayout.from_contracts())          # facts from the scanner and the data, no LLM
    files = render_team(plan)                       # path -> text, every file checked
    report = write_team(plan, files)                # new files only; existing ones are verified, never overwritten

The project already runs (Dockerfile, docker-compose), so nothing of it is rebuilt. The Setup Agent adds
what is missing: the Makefile (install / train / serve / test / all), the CI workflow (ruff + pytest),
the Pandera schema of a batch, the dependency-map seed and agents.yaml. Layout agreed with the team:
code in baseline/, data in data/ at the root, every command run from the root (roadmap 5.2).
"""

from __future__ import annotations

import ast
import csv
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ..decide.contracts import load_contracts
from ..render import MARKER, _env
from ..repo_writer import RepoWriter
from ..scan import scan_repo
from ..scan.deps import normalize
from ..scan.known import IMPORT_TO_DIST

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts.yaml"
SAMPLE_ROWS = 600   # roadmap 7.3 says 200; 600 because the Trainer's train.py holds out a fixed 400 rows (team choice, HANDOFF 14.13)


class TeamLayout(BaseModel):
    """Where things are in the team repo: contracts.yaml integration.layout (one source per value)."""
    root: str = "."
    code_dir: str
    data_path: str
    data_url: str
    config_sample: str                             # the hand-made sample the generated agents.yaml must match
    makefile: str
    ci: str
    schema_file: str
    team_schema: str = ""                          # the team's own schema module (Data Checker): used, never generated
    depmap: str
    agents_yaml: str
    tests_dir: str

    @classmethod
    def from_contracts(cls, contracts_path: str | Path = CONTRACTS, root: str = ".") -> "TeamLayout":
        c = load_contracts(contracts_path)
        return cls(root=root, **{name: c.i("layout", name) for name in cls.model_fields if name != "root"})

    @classmethod
    def for_repo(cls, root: str | Path, contracts_path: str | Path = CONTRACTS) -> "TeamLayout":
        """Any small ML repo (roadmap 7.1: the two test repos): the code is the whole repo, the Makefile at its root,
        the data the first CSV the scanner finds; the generated files keep the team's names."""
        team = cls.from_contracts(contracts_path)
        ctx = scan_repo(root)
        csvs = sorted((d for d in ctx.data_columns if d.path.endswith((".csv", ".tsv"))),
                      key=lambda d: (-(d.n_rows or 0), d.path))
        if not csvs:
            raise FileNotFoundError(f"no CSV data found in {root}: nothing to build a schema from")
        return team.model_copy(update={
            "root": str(root), "code_dir": ".", "data_path": csvs[0].path, "data_url": "", "team_schema": "",
            "makefile": Path(team.makefile).name,
            "config_sample": str(Path(contracts_path).resolve().parents[2] / team.config_sample)})

    @classmethod
    def detect(cls, root: str | Path = ".", contracts_path: str | Path = CONTRACTS) -> "TeamLayout":
        """The team layout when the repo has it (baseline/, and the wine data or the URL to download it: a fresh
        clone has no data/), else the layout of a small repo."""
        team = cls.from_contracts(contracts_path, root=str(root))
        if (Path(root) / team.code_dir).is_dir() and ((Path(root) / team.data_path).is_file() or team.data_url):
            return team
        return cls.for_repo(root, contracts_path)


class Finding(BaseModel):
    """Something the Setup Agent noticed and reports, without changing the team's code."""
    kind: str                                      # undeclared_dependency, no_model_output, ...
    path: str
    message: str
    owner: str | None = None                       # the agent whose file it is (roadmap 2.3)


def _owner_of(path: str) -> str | None:
    """Who should fix a finding in this file (roadmap 2.3): the serving app and its Dockerfile are the deployer's,
    training code the trainer's; None when unclear (the board decides)."""
    parts = Path(path).parts
    if "app" in parts or Path(path).name == "Dockerfile":
        return "deployer"
    if Path(path).name.startswith("train"):
        return "trainer"
    return None


class Column(BaseModel):
    raw: str                                       # CSV header ("fixed acidity")
    name: str                                      # feature name ("fixed_acidity")
    low: float
    high: float
    integer: bool


class TeamPlan(BaseModel):
    layout: TeamLayout
    python_version: str
    requirements: list[str]                        # requirement files to install, root-relative
    extra_packages: list[str]                      # used by the code but declared nowhere
    train_script: str | None                       # root-relative
    train_argv_data: bool                          # the script reads the data path from sys.argv[1]
    train_data_flag: str | None = None             # or from an option (argparse --data): given the path with it
    team_schema: str | None = None                 # the team's schema module when it has one: make test uses it
    train_style: str = "script"                    # script: python path/train.py; module: python -m pkg.train
    train_cwd: str | None = None                   # root-relative folder train runs in (None: the Makefile's)
    has_tests: bool = True                         # the tests folder exists (else make test runs only the schema)
    sample_rows: int = SAMPLE_ROWS                 # agents.yaml agents.setup.thresholds.dry_run_sample_rows
    team_tests: str | None = None                  # the team's own tests (contract, agents): a CI step, not make test
    decisions: list = Field(default_factory=list)  # the decide step: python, target, train_script (team/decide.py)
    target_check: str = "pa.Check.in_range(0, 10)"   # the target column's check, from the data (pandera code)
    target_type: str = "int"
    serve_target: str | None                       # module:app, importable from the root
    separator: str
    target: str
    columns: list[Column]
    computes: dict[str, list[str]]                 # code file -> features it names
    modes: dict[str, str] = Field(default_factory=dict)      # output path -> create | verify
    findings: list[Finding] = Field(default_factory=list)


def feature_name(raw: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", raw.strip()).strip("_")


def _separator(path: Path) -> str:
    head = path.read_text(encoding="utf-8", errors="replace")[:4096]
    try:
        return csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _columns(data: Path, sep: str, target: str) -> list[Column]:
    """Ranges as samples/schema.py: 0.1-99.9 percentiles +-10%, widened to cover every row of the data."""
    import pandas as pd
    df = pd.read_csv(data, sep=sep)
    out = []
    for raw in df.columns:
        if raw == target or not pd.api.types.is_numeric_dtype(df[raw]):
            continue
        s = df[raw].dropna()
        low = min(float(s.quantile(0.001)) * 0.9, float(s.min()))
        high = max(float(s.quantile(0.999)) * 1.1, float(s.max()))
        out.append(Column(raw=raw, name=feature_name(raw), low=round(low, 4), high=round(high, 4),
                          integer=pd.api.types.is_integer_dtype(df[raw])))
    return out


def _target_check(data: Path, sep: str, target: str) -> tuple[str, str]:
    """(dtype, pandera check) for the target, from the data like the features: a number gets its range
    (0.1-99.9 percentiles +-10%, widened to every row), a label its set of values."""
    import pandas as pd
    df = pd.read_csv(data, sep=sep, usecols=[target])
    s = df[target].dropna()
    if pd.api.types.is_numeric_dtype(s):
        low = min(float(s.quantile(0.001)) * 0.9, float(s.min()))
        high = max(float(s.quantile(0.999)) * 1.1, float(s.max()))
        if pd.api.types.is_integer_dtype(s):          # a score or a class number: whole bounds, rounded outward
            import math
            return "int", f"pa.Check.in_range({math.floor(low)}, {math.ceil(high)})"
        return "float", f"pa.Check.in_range({round(low, 4)}, {round(high, 4)})"
    values = sorted(s.astype(str).unique())
    return "str", f"pa.Check.isin({values!r})" if len(values) <= 50 else "pa.Check.str_length(1, 200)"


def _string_constants(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _top_level_train_scripts(code: Path, ctx) -> list:
    """The train*.py that fit a model in top-level code (no __main__ guard, no CLI): common in small repos, and run
    as a file they train. The scanner keeps script mode for guarded scripts; team mode accepts these too.
    All of them: with several, the decide step chooses (it used to take the first one and call it the only one)."""
    from ..scan import scripts
    from ..scan.models import TrainScript
    found = []
    for ep in sorted(ctx.entry_points, key=lambda e: e.path):
        if not Path(ep.path).name.startswith("train"):
            continue
        try:
            tree = ast.parse((code / ep.path).read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        if scripts.trains(tree):
            found.append(TrainScript(path=ep.path, has_main_guard=False, cli=None, cli_args=[],
                                     argv=scripts.argv_positions(tree), outputs=scripts.model_outputs(tree),
                                     imports_mlflow=any(isinstance(n, (ast.Import, ast.ImportFrom)) and "mlflow" in
                                                        ast.dump(n) for n in ast.walk(tree))))
    return found


SAVES_MODEL = re.compile(r"\b(pickle\.dumps?|joblib\.dump|\.save_model|torch\.save|mlflow\.\w+\.log_model)\(")


def _saves_model(script, code: Path) -> bool:
    return bool(script.outputs or script.imports_mlflow or SAVES_MODEL.search(
        (code / script.path).read_text(encoding="utf-8", errors="replace")))


def ensure_data(layout: TeamLayout, timeout: float = 120) -> str | None:
    """The data is not in git (roadmap 5.2): on a fresh clone, download it from layout.data_url to
    layout.data_path, as the generated `make data` does. Returns what was done, or None when it was there.
    Raises FileNotFoundError (with the reason) when it is missing and can't be downloaded."""
    import shutil
    import tempfile
    import urllib.request
    data = Path(layout.root).resolve() / layout.data_path
    if data.is_file():
        return None
    if not layout.data_url:
        raise FileNotFoundError(f"{layout.data_path} is missing and no data_url is set (contracts.yaml "
                                "integration.layout.data_url): add the data or the URL")
    data.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(layout.data_url, timeout=timeout) as r, \
                tempfile.NamedTemporaryFile(dir=data.parent, delete=False, suffix=".part") as tmp:
            shutil.copyfileobj(r, tmp)
        Path(tmp.name).replace(data)                # whole file or nothing
    except Exception as e:  # noqa: BLE001 - offline, 404: said plainly
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except NameError:
            pass
        raise FileNotFoundError(f"{layout.data_path} is missing and the download from {layout.data_url} failed "
                                f"({type(e).__name__}: {e})") from e
    return f"downloaded {layout.data_path} from {layout.data_url} (the data is not in git)"


def team_plan(layout: TeamLayout, answers: dict[str, str] | None = None) -> TeamPlan:
    """answers: the LLM's choices for the needs_llm decisions (resolve_with_llm); they replace the rules' guesses."""
    from .decide import decide_python, decide_target, decide_train_script
    answers = answers or {}
    root = Path(layout.root).resolve()
    code = root / layout.code_dir
    data = root / layout.data_path
    downloaded = ensure_data(layout)                 # a fresh clone has no data/: fetched like `make data`
    if downloaded:
        print(f"[setup] {downloaded}")
    ctx = scan_repo(code)
    rel = lambda p: p if layout.code_dir in ("", ".") else f"{layout.code_dir}/{p}"   # noqa: E731
    findings: list[Finding] = []

    decisions = [decide_python(ctx, CONTRACTS)]

    reqs = sorted(rel(d.path) for d in ctx.dependency_files if d.kind == "requirements")
    # the root requirements first, then the app's (the MLflow server image's own file is not ours to install)
    reqs = [r for r in reqs if not r.startswith(rel("mlflow/"))]
    declared = {normalize(p.name) for d in ctx.dependency_files if d.kind == "requirements"
                and not d.path.startswith("mlflow/") for p in d.packages}
    extra = []
    for u in ctx.undeclared_imports:
        dist = IMPORT_TO_DIST.get(u.distribution, u.distribution)
        if u.notebooks_only or normalize(dist) in declared:
            continue
        extra.append(dist)
        owners = {_owner_of(rel(f)) for f in u.files}
        findings.append(Finding(kind="undeclared_dependency", path=", ".join(rel(f) for f in u.files),
                                owner=owners.pop() if len(owners) == 1 else None,
                                message=f"{dist} is imported but no requirements file declares it; "
                                        f"the Makefile installs it until it is added"))

    for dockerfile in sorted(code.rglob("Dockerfile")):
        if "mlflow" in dockerfile.relative_to(code).parts:
            continue                                    # the MLflow server image, not the project's code
        for line in dockerfile.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"pip3? install\s+(.*)", line)
            if not m:
                continue
            for word in m.group(1).split():
                if word.startswith("-") or word.endswith(".txt") or word in ("&&", "\\"):
                    continue
                name = re.split(r"[=<>!~\[]", word, maxsplit=1)[0]
                if normalize(name) in declared or name in extra:
                    continue
                extra.append(word)
                findings.append(Finding(kind="undeclared_dependency", path=dockerfile.relative_to(root).as_posix(),
                                        owner=_owner_of(dockerfile.relative_to(root).as_posix()),
                                        message=f"{word} is installed only in this Dockerfile, not in a requirements "
                                                f"file; the Makefile installs it too"))

    # the MLflow client must match the tracking server (fault-001: a 3.x client calls endpoints a 2.x server lacks)
    server_pin = None
    for f in sorted(code.rglob("requirements*.txt")):
        if "mlflow" in f.relative_to(code).parts[:-1]:
            m = re.search(r"^mlflow==([\w.]+)", f.read_text(encoding="utf-8", errors="replace"), re.M)
            server_pin = (m.group(1), f.relative_to(root).as_posix()) if m else server_pin
    if server_pin:
        for rel_req in reqs:
            m = re.search(r"^mlflow==([\w.]+)", (root / rel_req).read_text(encoding="utf-8", errors="replace"), re.M)
            if m and m.group(1) != server_pin[0]:
                findings.append(Finding(kind="version_mismatch", path=rel_req, owner="deployer",
                                        message=f"pins mlflow=={m.group(1)} but the tracking server runs "
                                                f"mlflow=={server_pin[0]} ({server_pin[1]}); the client should match "
                                                f"the server (roadmap 5.3)"))

    scripts = {s.path: s for s in ctx.train_scripts} or {s.path: s for s in _top_level_train_scripts(code, ctx)}
    decisions.append(decide_train_script(sorted(scripts), code,
                                         saves={p for p, s in scripts.items() if _saves_model(s, code)}))
    script = scripts.get(answers.get("train_script") or decisions[-1].value)
    saves = script and _saves_model(script, code)
    if script and not saves:
        findings.append(Finding(kind="no_model_output", path=rel(script.path), owner="trainer",
                                message="the training script saves no model (no MLflow log, no file): make train "
                                        "runs, but nothing can be served from it (the trainer's file)"))
    app = ctx.web_apps[0] if ctx.web_apps else None
    serve = (app.target if layout.code_dir in ("", ".") else f"{layout.code_dir.replace('/', '.')}.{app.target}") \
        if app else None

    sep = _separator(data)
    with open(data, encoding="utf-8", errors="replace") as f:
        header = next(csv.reader([f.readline()], delimiter=sep))
    decisions.append(decide_target(ctx, header, script and (
        script.path, (code / script.path).read_text(encoding="utf-8", errors="replace"))))
    for d in decisions:
        if d.item in answers and d.status == "needs_llm":
            d.value, d.source = answers[d.item], "llm"
    target = next(d.value for d in decisions if d.item == "target")
    python_version = next(d.value for d in decisions if d.item == "python") or "3.10"
    columns = _columns(data, sep, target)
    target_type, target_check = _target_check(data, sep, target)
    features = {c.raw for c in columns} | {c.name for c in columns}
    computes = {}
    for ep in ctx.entry_points:
        named = _string_constants(code / ep.path) & features
        if named:
            computes[rel(ep.path)] = sorted({feature_name(n) for n in named},
                                            key=[c.name for c in columns].index)

    plan = TeamPlan(layout=layout, python_version=python_version, requirements=reqs, extra_packages=extra,
                    decisions=[d.model_dump() for d in decisions], target_type=target_type,
                    target_check=target_check,
                    train_script=rel(script.path) if script else None,
                    train_argv_data=bool(script and 1 in script.argv), serve_target=serve, separator=sep,
                    has_tests=(root / layout.tests_dir).is_dir(),
                    team_tests="tests" if (root / "tests").is_dir() and (root / "requirements.txt").is_file()
                    and layout.tests_dir != "tests" else None,
                    target=target, columns=columns, computes=computes, findings=findings)
    plan.train_data_flag = data_flag(script) if script and not plan.train_argv_data else None
    plan.team_schema = team_schema_of(layout, root)
    plan.modes = {p: ("verify" if (root / p).exists() else "create") for p in outputs(layout)
                  if not (plan.team_schema and p == layout.schema_file)}      # the team's schema replaces ours
    return plan


DATA_FLAGS = {"data", "data_path", "datapath", "dataset", "input", "input_path", "csv", "data_file", "train_data",
              "train_path"}


def data_flag(script) -> str | None:
    """The command-line option a training script takes its data path from (`--data`, `--data-path`, ...)."""
    for arg in getattr(script, "cli_args", None) or []:
        for flag in arg.flags:
            if flag.startswith("--") and flag.lstrip("-").replace("-", "_").lower() in DATA_FLAGS:
                return flag
    return None


def team_schema_of(layout: TeamLayout, root: Path) -> str | None:
    """The team's schema module (layout.team_schema) when it exists and defines a pandera DataFrameSchema."""
    if not layout.team_schema:
        return None
    path = root / layout.team_schema
    if path.is_file() and "DataFrameSchema" in path.read_text(encoding="utf-8", errors="replace"):
        return layout.team_schema
    return None


def outputs(layout: TeamLayout) -> list[str]:
    return [layout.makefile, layout.ci, layout.schema_file, layout.depmap, layout.agents_yaml]


# --- rendering -----------------------------------------------------------------------------------

def _agents_yaml(plan: TeamPlan) -> str:
    root = Path(plan.layout.root).resolve()
    cfg = yaml.safe_load((root / plan.layout.config_sample).read_text(encoding="utf-8"))
    cfg["features"] = [c.name for c in plan.columns]
    cfg["data"] = {"path": plan.layout.data_path, "separator": plan.separator, "target": plan.target}
    cfg["schema_path"] = plan.team_schema or plan.layout.schema_file
    cfg["dependency_map"] = plan.layout.depmap
    head = (f"{MARKER}\n# {plan.layout.agents_yaml}, generated by the Setup Agent from {plan.layout.config_sample} and the scanned "
            f"project.\n# The board owns it from now on (roadmap 5.13).\n")
    return head + yaml.safe_dump(cfg, sort_keys=False, width=110, allow_unicode=True)


def resolve_with_llm(plan: TeamPlan, client) -> tuple[TeamPlan, list[dict]]:
    """The needs_llm decisions go to the LLM; a valid answer re-plans with it (the target changes the schema's
    columns). No LLM, or no valid answer: the rules' guesses stay, marked "assumed" and reported as findings."""
    from .decide import Decision, ask_llm
    decisions = [Decision(**d) for d in plan.decisions]
    if not any(d.status == "needs_llm" for d in decisions):
        return plan, []
    log = []
    answers = {}
    if client is not None:
        facts = "\n".join([f"code: {plan.layout.code_dir}/, data: {plan.layout.data_path}",
                           *(f"finding: {f.message}" for f in plan.findings)])
        answers, log = ask_llm(decisions, client, facts)
    new = team_plan(plan.layout, answers) if answers else plan
    for d in new.decisions:
        if d["status"] == "needs_llm" and d["source"] != "llm":
            d["source"] = "assumed"
            new.findings.append(Finding(kind="decision_assumed", path=plan.layout.code_dir,
                                        message=f"{d['item']}: {d['reason']}; no LLM answer, so the rules' best "
                                                f"guess {d['value']!r} is used (check it)"))
    return new, log


def make_dir(layout: TeamLayout) -> str:
    """The folder of the pipeline Makefile, root-relative ("baseline"; "." for a root Makefile)."""
    return Path(layout.makefile).parent.as_posix()


def from_make_dir(layout: TeamLayout, path: str) -> str:
    """A root-relative path as seen from the Makefile's folder (make -C <folder> runs there)."""
    return os.path.relpath(path, make_dir(layout)).replace("\\", "/")


def module_of(script: str, root: Path) -> tuple[str, str] | None:
    """(folder to put on the path, dotted module) for a script inside a package: src/irisclf/train.py ->
    ("src", "irisclf.train"). None when the script's folder is not a package (no __init__.py)."""
    path = Path(script)
    parts, folder = [path.stem], path.parent
    while (root / folder / "__init__.py").is_file():
        parts.insert(0, folder.name)
        folder = folder.parent
    return (folder.as_posix(), ".".join(parts)) if len(parts) > 1 else None


def train_command(plan: TeamPlan) -> str:
    """The train recipe, run from the Makefile's folder. Data paths are made absolute so a change of folder
    (fix_path workdir) or of style (change_entry_command module) keeps them right."""
    lay, root = plan.layout, Path(plan.layout.root)
    data = (f" {plan.train_data_flag} $(abspath $(RUN_DATA))" if plan.train_data_flag else
            " $(abspath $(RUN_DATA))" if plan.train_argv_data else "")
    if plan.train_style == "module" and (mod := module_of(plan.train_script, root)):
        path_dir, module = mod
        cwd = plan.train_cwd or "."                               # modules run from the root by default
        return (f"cd {from_make_dir(lay, cwd)} && PYTHONPATH={os.path.relpath(path_dir, cwd)} "
                f"$(PYTHON) -m {module}{data}").replace("\\", "/")
    if plan.train_cwd is not None:
        return (f"cd {from_make_dir(lay, plan.train_cwd)} && "
                f"$(PYTHON) {os.path.relpath(plan.train_script, plan.train_cwd)}{data}").replace("\\", "/")
    return f"$(PYTHON) {from_make_dir(lay, plan.train_script)}{data}"


def serve_module(plan: TeamPlan) -> str | None:
    """module:app importable from the Makefile's folder (baseline.app.main:app -> app.main:app)."""
    if not plan.serve_target:
        return None
    prefix = make_dir(plan.layout).replace("/", ".")
    module, _, obj = plan.serve_target.partition(":")
    if prefix != "." and module.startswith(prefix + "."):
        module = module[len(prefix) + 1:]
    return f"{module}:{obj}"


def render_team(plan: TeamPlan) -> dict[str, str]:
    env = _env()
    env.filters["here"] = lambda p: from_make_dir(plan.layout, p)
    ctx = {"plan": plan, "layout": plan.layout, "marker": MARKER, "sample_rows": plan.sample_rows,
           "data_dir": str(Path(plan.layout.data_path).parent).replace("\\", "/"), "data_url": plan.layout.data_url,
           "make_dir": make_dir(plan.layout), "serve_module": serve_module(plan),
           "train_command": train_command(plan) if plan.train_script else None}
    files = {
        plan.layout.makefile: env.get_template("team_makefile.j2").render(**ctx),
        plan.layout.ci: env.get_template("team_ci.yml.j2").render(**ctx),
        plan.layout.schema_file: env.get_template("team_schema.j2").render(**ctx),
        plan.layout.depmap: env.get_template("team_depmap.yaml.j2").render(**ctx),
        plan.layout.agents_yaml: _agents_yaml(plan),
    }
    if plan.team_schema:                     # the Data Checker owns the schema: no second one that could disagree
        files.pop(plan.layout.schema_file)
    problems = check_files(plan, files)
    if problems:
        raise ValueError("generated files don't check: " + "; ".join(problems))
    return files


# --- checks --------------------------------------------------------------------------------------

def check_files(plan: TeamPlan, files: dict[str, str]) -> list[str]:
    """Every generated file parses before anything runs: YAML loads, Python compiles, make reads the Makefile."""
    out = []
    lay = plan.layout
    for path in (lay.ci, lay.depmap, lay.agents_yaml):
        try:
            data = yaml.safe_load(files[path])
        except yaml.YAMLError as e:
            out.append(f"{path}: {e}")
            continue
        if path == lay.depmap and not ({"owners", "links"} <= set(data or {})):
            out.append(f"{path}: needs owners and links")
    if lay.schema_file in files:
        try:
            compile(files[lay.schema_file], lay.schema_file, "exec")
        except SyntaxError as e:
            out.append(f"{lay.schema_file}: {e}")
    sample = yaml.safe_load((Path(lay.root).resolve() / lay.config_sample).read_text(encoding="utf-8"))
    generated = yaml.safe_load(files[lay.agents_yaml])
    missing = sorted(set(sample) - set(generated)) + sorted(f"agents.{a}" for a in set(sample.get("agents", {}))
                                                          - set(generated.get("agents", {})))
    if missing:
        out.append(f"{lay.agents_yaml}: keys of the sample missing: {', '.join(missing)}")
    if shutil.which("make"):
        r = subprocess.run(["make", "-n", "-f", "-", "all"], input=files[lay.makefile], capture_output=True,
                           text=True, cwd=Path(lay.root) / make_dir(lay))
        if r.returncode != 0:
            out.append(f"{lay.makefile}: make -n all fails: {r.stderr.strip()[:300]}")
    return out


# --- writing -------------------------------------------------------------------------------------

class TeamReport(BaseModel):
    created: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)            # the agent's own files rewritten on a rebuild
    verified: dict[str, str] = Field(default_factory=dict)     # path -> "same" or what differs
    findings: list[Finding] = Field(default_factory=list)


def _digest(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode()).hexdigest()


def write_team(plan: TeamPlan, files: dict[str, str], root: str | Path | None = None,
               update_own: bool = False) -> TeamReport:
    """Write the files that don't exist (in `root`, default the project); compare the ones that do. Never
    overwrites a team file: a difference is reported, not fixed. update_own (request.rebuild): a file this agent
    generated (its first line is the marker) is rewritten when it differs, but only the Makefile and the CI
    workflow: agents.yaml (the board), schema.py (the Checker) and the dependency map (Reviewer, Trainer) are a
    first draft that others own (contracts.yaml ownership). Writes go through RepoWriter, like
    every write into a target repo (in a git repo: on the builder/<date> branch)."""
    base = Path(root or plan.layout.root).resolve()
    report = TeamReport(findings=list(plan.findings))
    new, own = {}, {}
    for path, text in files.items():
        target = base / path
        if target.exists():
            current = target.read_text(encoding="utf-8", errors="replace")
            same = _digest(current) == _digest(text)
            mine = current.replace("\r\n", "\n").split("\n", 1)[0].strip() == MARKER.strip()
            if not same and mine and update_own and path in (plan.layout.makefile, plan.layout.ci):
                own[path] = text
                report.verified[path] = "updated (generated by this agent)"
            else:
                report.verified[path] = "same" if same else "differs from what the Setup Agent would generate"
        else:
            new[path] = text
    writer = RepoWriter(base) if (new or own) else None
    if own:
        report.updated = writer.update_own(own, MARKER)
    if new:
        report.created = writer.write_new(new)
    return report
