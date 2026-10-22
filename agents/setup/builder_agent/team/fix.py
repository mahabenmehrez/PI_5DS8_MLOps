"""Team mode's LLM step: when no rule fixes a failed dry run, the LLM picks one fix from a small menu.

Same machinery as the Builder's fix loop (builder_agent.fix.diagnose): one-shot prompt with the error, the
project facts and the similar verified fixes from memory (hybrid RAG), a strict JSON schema, and a validator
that sends rejected answers back with the reasons (at most 3 calls). The menu is smaller: there is no Docker
image here, so a fix can only change what `make install` installs.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from typing import Annotated, Literal, Union

from pydantic import Field

from ..fix import MENU, diagnose
from ..fix.models import AddDependency, GiveUp, PinPackage, _Strict, describe
from ..llm.prompts import load_prompt
from ..memory import render_retrieved
from ..render.configs import Overrides
from ..scan.deps import normalize

class FixPath(_Strict):
    """Run train from another folder: for a script that opens files relative to its own folder."""
    action: Literal["fix_path"]
    target: Literal["workdir"]
    path: str


class ChangeEntryCommand(_Strict):
    """Run train as a file (python src/pkg/train.py) or as a module (python -m pkg.train, for relative imports)."""
    action: Literal["change_entry_command"]
    stage: Literal["train"]
    style: Literal["script", "module"]


TeamFix = Annotated[Union[PinPackage, AddDependency, FixPath, ChangeEntryCommand, GiveUp], Field(discriminator="action")]


class TeamAnswer(_Strict):
    fix: TeamFix
    reason: str = Field(max_length=400)


TEAM_MENU = {
    "add_dependency": MENU["add_dependency"],
    "pin_package": MENU["pin_package"],
    "fix_path": "{target: \"workdir\", path}: run the training step from this folder of the repo (for a script "
                "that opens files relative to its own folder)",
    "change_entry_command": "{stage: \"train\", style: \"script\" | \"module\"}: run the training script as a file, "
                            "or as a module with python -m (needed for relative imports inside a package)",
    "give_up": MENU["give_up"],
}

PROMPT = "diagnose_team_v4"           # v1: plain; v2: + CRAG; v3: + installed/PyPI versions; v4: + where a missing file is
TEAM_ACTIONS = tuple(TEAM_MENU)
STAGE_ORDER = ["install", "data", "sample", "train", "test"]      # the order `make all SAMPLE=1` runs them in


def _name(spec: str) -> str:
    return normalize(spec.split("==")[0])


def installed(plan) -> set[str]:
    """Pip names make install installs: the requirement files' and the extra packages."""
    from pathlib import Path

    from ..scan.deps import parse_requirement
    names = set()
    for rel in plan.requirements:
        for line in (Path(plan.layout.root) / rel).read_text(encoding="utf-8", errors="replace").splitlines():
            r = parse_requirement(line)
            if r is not None:
                names.add(normalize(r.name))
    return names | {_name(p) for p in plan.extra_packages}


# "cannot import name 'soft_unicode' from 'markupsafe'": markupsafe IS installed (Python found it), the name is
# missing from its installed version, so the fix is a version (pin_package), never add_dependency
_FROM_MODULE = re.compile(r"cannot import name '?[\w.]+'? from '([\w.]+)'")


def present_in_error(log: str) -> set[str]:
    """Pip names the error shows are installed (the module was found, something inside it is missing)."""
    from ..scan.known import IMPORT_TO_DIST
    return {normalize(IMPORT_TO_DIST.get(m.split(".")[0], m.split(".")[0])) for m in _FROM_MODULE.findall(log)}


def overrides(plan) -> Overrides:
    """The fixes already in the plan, in the form the validator reads (pins and added dependencies)."""
    pins = {p.split("==")[0]: p.split("==")[1] for p in plan.extra_packages if "==" in p}
    return Overrides(pins=pins, dependencies=[p for p in plan.extra_packages if "==" not in p])


def known_packages(plan, log: str = "") -> set[str]:
    """Package names a fix may install: named in the error output, already installed by the project, or the pip
    name of an import the agent knows. Anything else could be a made-up or malicious name (supply chain): pip
    install runs its code, so a name from the LLM (or from a web page it was shown) is never trusted alone."""
    from ..scan.known import FRAMEWORKS, IMPORT_TO_DIST
    words = set(re.findall(r"[A-Za-z][A-Za-z0-9_.\-]+", log))
    names = {normalize(w) for w in words} | {normalize(IMPORT_TO_DIST.get(w, w)) for w in words}
    return names | installed(plan) | {normalize(d) for d in IMPORT_TO_DIST.values()} | {normalize(f) for f in FRAMEWORKS}


# ---- v2: what the dry run really installed, and what really exists on PyPI ----------------------------------------

PYPI_JSON = "https://pypi.org/pypi/{}/json"
_RELEASES: dict[str, list[str] | None] = {}
NOT_PACKAGES = {"pip", "setuptools", "wheel"}
# pip's own lines in the log: they name every installed package, so they never count as "the error names it"
_PIP_LINE = re.compile(r"^\s*(Requirement already satisfied|Collecting|Downloading|Using cached|Installing collected|"
                       r"Successfully|Attempting uninstall|Found existing|Uninstalling|Building wheel|Created wheel|"
                       r"Stored in|Preparing metadata|Obtaining|Looking in)")


def pypi_releases(name: str, timeout: float = 10, python: str | None = None) -> list[str] | None:
    """The final (not pre-release, not yanked) versions of a package on PyPI, oldest first; None when PyPI can't be
    reached (then the check is skipped, never guessed). Cached for the process.
    python ("3.10.14"): only the versions that install on it (requires_python of their files, v4)."""
    key = normalize(name)
    if key not in _RELEASES:
        import json
        import urllib.request

        from packaging.version import InvalidVersion, Version
        try:
            with urllib.request.urlopen(PYPI_JSON.format(key), timeout=timeout) as r:
                releases = json.load(r).get("releases", {})
            good = []
            for v, files in releases.items():
                try:
                    pv = Version(v)
                except InvalidVersion:
                    continue
                live = [f for f in files if not f.get("yanked")]
                if live and not pv.is_prerelease:
                    good.append((pv, v, [f.get("requires_python") for f in live]))
            _RELEASES[key] = [(v, req) for _, v, req in sorted(good)]
        except Exception:  # noqa: BLE001 - offline, 404: unknown
            _RELEASES[key] = None
    if _RELEASES[key] is None:
        return None
    return [v for v, req in _RELEASES[key] if python is None or any(_python_ok(r, python) for r in req)]


def _python_ok(requires: str | None, python: str) -> bool:
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    if not requires:
        return True
    try:
        return SpecifierSet(requires).contains(python, prereleases=True)
    except InvalidSpecifier:
        return True                       # unreadable metadata: not a reason to refuse


def latest_per_minor(releases: list[str], n: int = 10) -> list[str]:
    """The newest patch of each minor version, newest first (1.7.2, 1.6.1, 1.5.2, ...)."""
    from packaging.version import Version
    best: dict[tuple, str] = {}
    for v in releases:
        pv = Version(v)
        best[pv.release[:2]] = v
    return [best[k] for k in sorted(best, reverse=True)][:n]


def error_text(log: str) -> str:
    """The failed stage's last lines without pip's install lines."""
    return "\n".join(line for line in log.splitlines()[-50:] if not _PIP_LINE.match(line))


def packages_in_error(log: str, versions: dict[str, str]) -> list[str]:
    """Installed pip names the error output names, by pip name or import name (sklearn -> scikit-learn)."""
    from ..scan.known import IMPORT_TO_DIST
    text = error_text(log)
    imports: dict[str, set[str]] = {}
    for imp, dist in IMPORT_TO_DIST.items():
        imports.setdefault(normalize(dist), set()).add(imp)
    found = []
    for dist in versions:
        d = normalize(dist)
        if d in NOT_PACKAGES:
            continue
        names = {dist, d, d.replace("-", "_")} | imports.get(d, set())
        if any(re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", text, re.I) for n in names):
            found.append(d)
    return sorted(found)


def versions_text(result: dict) -> str:
    """For the prompt: the installed version and the real PyPI versions of each package the error names, and what
    the import check found. The LLM picks a version from a real list instead of from its memory."""
    versions = {normalize(k): v for k, v in (result.get("installed") or {}).items()}
    lines = []
    python = result.get("python")
    for d in packages_in_error(result.get("log", ""), versions):
        rel = pypi_releases(d, python=python)
        avail = ", ".join(latest_per_minor(rel)) if rel else "unknown (PyPI not reachable)"
        where = f" that install on Python {python}" if python else ""
        lines.append(f"- {d}: installed {versions[d]}; on PyPI{where} (newest patch of each minor, newest first): "
                     f"{avail}")
    hint = result.get("import_hint")
    if hint and hint.get("close"):
        lines.append(f"- import check: {hint['module']} has {', '.join(hint['close'])}, close to the name the code "
                     f"imports ({hint['name']!r}): the project's code misspells it, and no version has the misspelt name")
    return "\n".join(lines) or "none of the installed packages is named in the error output"


# ---- v3: a missing file ------------------------------------------------------------------------------------------

SKIP_DIRS = {".git", ".setup", ".venv", "venv", "__pycache__", "node_modules", "mlruns", "mlartifacts", ".cache",
             ".tools", "logs", ".pytest_cache"}


def missing_file(result: dict) -> str | None:
    """The relative path the failed stage couldn't open (FileNotFoundError), or None. An absolute path (a file
    inside an installed package, say) is not the project's path problem."""
    detail = result.get("detail")
    if result.get("error_class") == "FileNotFoundError" and detail and not detail.startswith(("/", "\\")) \
            and ":" not in detail[:3]:
        return detail.replace("\\", "/")
    return None


def files_named(root, name: str, limit: int = 5) -> list[str]:
    """Repo-relative paths of the files called `name` (the agent's own folders left out)."""
    from pathlib import Path
    root = Path(root)
    out = []
    for p in sorted(root.rglob(name)):
        rel = p.relative_to(root)
        if p.is_file() and not SKIP_DIRS & set(rel.parts[:-1]):
            out.append(rel.as_posix())
            if len(out) >= limit:
                break
    return out


def where_text(plan, result: dict) -> str:
    """For the prompt: where the missing file really is, and from which folder the code's path would open it."""
    from pathlib import PurePosixPath
    detail = missing_file(result)
    if detail is None:
        return "the error is not a missing file"
    here = current_cwd(plan)
    lines = [f"- the code opened {detail!r} from {here}/, where it does not exist"]
    found = files_named(plan.layout.root, PurePosixPath(detail).name)
    if not found:
        lines.append(f"- no file named {PurePosixPath(detail).name!r} exists in the repository: the data is missing "
                     "or the code names the wrong file; that is the project's to fix (give_up)")
    for f in found:
        if ".." not in PurePosixPath(detail).parts and (f == detail or f.endswith("/" + detail)):
            folder = f[: -len(detail)].rstrip("/") or "."
            lines.append(f"- found at {f}: run from {folder}/, the code's path {detail!r} opens it")
        else:
            lines.append(f"- found at {f}")
    return "\n".join(lines)


def fix_key(fix) -> tuple:
    """What makes two fixes the same fix (normalized names, folders without slashes)."""
    f = fix if isinstance(fix, dict) else fix.model_dump()
    a = f.get("action")
    if a in ("add_dependency", "pin_package"):
        return (a, normalize(f.get("name", "")), f.get("version"))
    if a == "fix_path":
        return (a, (f.get("path") or "").strip("/") or ".")
    if a == "change_entry_command":
        return (a, f.get("style"))
    return (a,)


def current_cwd(plan) -> str:
    """The root-relative folder train runs in now (module style: the root; script style: the Makefile's folder)."""
    from . import make_dir
    if plan.train_cwd is not None:
        return plan.train_cwd.strip("/") or "."
    return "." if plan.train_style == "module" else (make_dir(plan.layout) or ".")


def team_checks(plan, log: str = "", result: dict | None = None, tried=()):
    """The validator's checks for the team-only actions: they must name something real and change something.
    v2 (from the v1 evaluation's failures): never a fix already tried; a pin must change the installed version,
    exist on PyPI and be of a package the error names; no version change for a misspelt name."""
    from pathlib import Path

    from . import module_of
    from ..scan.known import IMPORT_TO_DIST
    root = Path(plan.layout.root)
    allowed = known_packages(plan, log)
    result = result or {}
    versions = {normalize(k): v for k, v in (result.get("installed") or {}).items()}
    in_error = set(packages_in_error(log, versions)) | present_in_error(log)
    tried_keys = {fix_key(t) for t in tried}
    hint = result.get("import_hint") or {}
    misspelt_in = normalize(IMPORT_TO_DIST.get(hint.get("module", "").split(".")[0], hint.get("module", "").split(".")[0])) \
        if hint.get("close") else None

    missing = missing_file(result)
    python = result.get("python")

    def check(fix) -> list[str]:
        if missing and isinstance(fix, (AddDependency, PinPackage)):
            return [f"{fix.action}: the error is a missing file ({missing!r}); installing or pinning a package doesn't "
                    "create it. See WHERE THE MISSING FILE IS: fix_path to that folder, or give_up"]
        if fix_key(fix) in tried_keys:
            return [f"{describe(fix)} was already tried and the build still failed: choose a different fix"]
        if isinstance(fix, (AddDependency, PinPackage)) and normalize(fix.name) not in allowed:
            return [f"{fix.action}: {fix.name!r} is not named in the error output, not installed by the project and "
                    "not a known import: name the package the error is about"]
        if isinstance(fix, AddDependency) and normalize(fix.name) in versions:
            return [f"add_dependency: {fix.name} is already installed ({versions[normalize(fix.name)]})"]
        if isinstance(fix, PinPackage):
            name = normalize(fix.name)
            if misspelt_in and name == misspelt_in:
                return [f"pin_package: {hint['module']} has {', '.join(hint['close'])}: the code misspells "
                        f"{hint['name']!r}, no version of {fix.name} has that name. A misspelling in the project's "
                        "code is not the agent's to fix: give_up and say which file has it"]
            if versions.get(name) == fix.version:
                return [f"pin_package: {fix.name}=={fix.version} is the version the failed build already used: "
                        "pin a different version (see AVAILABLE VERSIONS)"]
            if versions and name not in in_error:
                return [f"pin_package: {fix.name} is not named in the error output: pin the package the error "
                        "comes from"]
            rel = pypi_releases(name)
            if rel is not None and fix.version not in rel:
                return [f"pin_package: {fix.name}=={fix.version} does not exist on PyPI; real versions: "
                        f"{', '.join(latest_per_minor(rel))}"]
            ok = pypi_releases(name, python=python) if python else None
            if rel is not None and ok is not None and fix.version not in ok:
                return [f"pin_package: {fix.name}=={fix.version} does not install on Python {python} (the build's "
                        f"Python); versions that do: {', '.join(latest_per_minor(ok))}"]
        if isinstance(fix, FixPath):
            p = fix.path.strip("/") or "."
            if ".." in Path(p).parts or not (root / p).is_dir():
                return [f"fix_path: {fix.path!r} is not a folder of the repository"]
            if p == current_cwd(plan):
                return [f"fix_path: train already runs in {p!r}"]
            if missing and not (root / p / missing).exists():
                return [f"fix_path: from {p}/ the code's path {missing!r} still doesn't exist (see WHERE THE "
                        "MISSING FILE IS)"]
        if isinstance(fix, ChangeEntryCommand):
            if fix.style == plan.train_style:
                return [f"change_entry_command: train already runs as a {fix.style}"]
            if fix.style == "module" and not (plan.train_script and module_of(plan.train_script, root)):
                return [f"change_entry_command: {plan.train_script} is not inside a package (no __init__.py), "
                        "so it can't run as a module"]
        return []
    return check


def apply(plan, fix) -> None:
    if isinstance(fix, FixPath):
        plan.train_cwd = fix.path.strip("/") or "."
    elif isinstance(fix, ChangeEntryCommand):
        plan.train_style = fix.style
    elif isinstance(fix, AddDependency):
        plan.extra_packages.append(fix.name)
    elif isinstance(fix, PinPackage):
        plan.extra_packages[:] = [p for p in plan.extra_packages if _name(p) != normalize(fix.name)]
        plan.extra_packages.append(f"{fix.name}=={fix.version}")


def regressed(before: dict, after: dict) -> bool:
    """The fix made an earlier stage fail than the one it was meant to fix."""
    if after["ok"] or before["ok"]:
        return False
    order = {s: n for n, s in enumerate(STAGE_ORDER)}
    return order.get(after["stage"], 99) < order.get(before["stage"], 99)


class TeamDiagnosis(BaseModel):
    fix: dict | None                  # the chosen fix (None: no valid answer, or the LLM was unreachable)
    text: str                         # describe(fix), or why there is none
    reason: str | None = None         # the LLM's reason
    give_up: bool = False
    calls: int = 0
    rejected: list[list[str]] = []    # validator reasons of the rejected answers
    retrieved: list[str] = []         # memory case ids shown to the LLM
    crag: dict | None = None          # Corrective RAG: action, grades, rewritten query, web sources
    call_log: list[dict] = []         # every LLM call: messages, answer, latency, tokens, valid, reasons (DoD 13.1.6)
    model: str | None = None


def diagnose_team(plan, result: dict, client, history: list[str], attempt: int, max_attempts: int,
                  memory=None, reverted: list[dict] = (), scan_summary: str = "", crag=None,
                  prompt: str | None = None, tried=()) -> tuple[TeamDiagnosis, object]:
    """One diagnosis. Returns (what was decided, the fix object or None).
    crag: a CorrectiveRAG (memory/crag.py): the retrieved cases are graded, refined, and completed by a web
    search when they don't fit. Without it: the advanced retriever's top 3 as they are (diagnose_team_v1)."""
    tail = result.get("log", "").splitlines()[-50:]
    crag_result = crag.retrieve(result["stage"], tail) if crag is not None else None
    hits = [] if crag is not None else (memory.advanced(result["stage"], tail) if memory is not None else [])
    variables = {
        "stage": result["stage"], "error_tail": "\n".join(tail), "scan_summary": scan_summary,
        "attempt": attempt, "max_attempts": max_attempts,
        "previous_fixes_text": "\n".join(f"- {h}" for h in history) or "none",
        "menu_text": "\n".join(f"- {k} {v}" for k, v in TEAM_MENU.items()),
        "pipeline_text": "\n".join([*(f"-r {r}" for r in plan.requirements), *plan.extra_packages,
                                    f"train runs as a {plan.train_style} ({plan.train_script}) from "
                                    f"{current_cwd(plan)}/"]),
        "retrieved_text": crag_result.render() if crag_result else render_retrieved(hits),
        "crag_action": (f"Corrective RAG, {crag_result.action}" if crag_result else
                        "Advanced RAG, not graded" if memory is not None else "none, no memory"),
        "versions_text": versions_text(result),
        "where_text": where_text(plan, result),
    }
    d, calls = diagnose(load_prompt(prompt or PROMPT), client, variables,
                        overrides(plan),
                        installed(plan) | present_in_error(result.get("log", "")),
                        reverted=list(reverted), allowed=set(TEAM_ACTIONS), answer=TeamAnswer,
                        extra_check=team_checks(plan, result.get("log", ""), result, tried))
    out = TeamDiagnosis(fix=None, text="", calls=len(calls),
                        retrieved=[g.id for g in crag_result.kept()] if crag_result else [h.doc.id for h in hits],
                        rejected=[c.reasons for c in calls if not c.valid],
                        call_log=[c.model_dump() for c in calls], model=getattr(client, "model", None),
                        crag=None if crag_result is None else {
                            "action": crag_result.action, "query": crag_result.query,
                            "web_source": crag_result.web_source,
                            "graded": [{"id": g.id, "score": g.score, "kept": g.kept} for g in crag_result.graded],
                            "web_urls": sorted({s["url"] for s in crag_result.web})})
    if d is None:
        errors = [c.error for c in calls if getattr(c, "error", None)]
        out.text = f"no valid answer from the LLM ({errors[-1] if errors else 'all answers rejected'})"
        return out, None
    out.fix, out.text, out.reason = d.fix.model_dump(), describe(d.fix), d.reason
    out.give_up = isinstance(d.fix, GiveUp)
    return out, (None if out.give_up else d.fix)
