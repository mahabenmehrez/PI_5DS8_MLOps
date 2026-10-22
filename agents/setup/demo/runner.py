"""What the demo page runs: the real Setup Agent, on a copy of the project, with every step reported live.

Nothing is precomputed: each run copies the project to a temporary folder (the real one is never written),
optionally adds one error to its training script, and calls SetupAgent.step(). The agent is the normal one;
the demo only listens: its dry runs, its rules, the decide step and each diagnosis (with CRAG) are reported as
events the page shows while the run goes on. Each finished run is saved in demo/recordings/, so the page can
replay it if something fails in front of an audience (the page says "replay" when it does).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agents.setup.demo.faults import BY_ID, inject

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                      # the team repo the demo runs from
RECORDINGS = HERE / "recordings"
ORDER = ["install", "data", "sample", "train", "tools", "test"]     # make all SAMPLE=1, in the order make runs it
STAGE_LABEL = {"install": "install", "data": "data", "sample": "sample", "train": "train", "tools": "tools",
               "test": "test"}

_busy = threading.Lock()                    # one run at a time: they share the dry-run venv cache
JOBS: dict[str, "Job"] = {}


class Busy(RuntimeError):
    pass


class Job:
    def __init__(self, action: str, fault: str | None, mode: str):
        self.id = uuid.uuid4().hex[:10]
        self.action, self.fault, self.mode = action, fault, mode
        self.events: list[dict] = []
        self.done = False
        self.result: dict | None = None
        self.start = time.time()
        self.builds = 0
        self.first_ok: bool | None = None

    def emit(self, event: str, **data) -> None:
        self.events.append({"t": round(time.time() - self.start, 1), "type": event, **data})

    def view(self, since: int = 0) -> dict:
        return {"id": self.id, "action": self.action, "fault": self.fault, "mode": self.mode, "done": self.done,
                "events": self.events[since:], "next": len(self.events), "result": self.result if self.done else None,
                "elapsed": round(time.time() - self.start, 1)}


# ---------------------------------------------------------------- what the page shows of the agent's settings

def status() -> dict:
    """Which models the agent would use, and whether they answer (no key value is ever returned)."""
    from agents.setup.builder_agent.llm.api import load_dotenv
    load_dotenv()
    out: dict = {"make": bool(shutil.which("make")), "recordings": recordings()}
    cfg = _config()
    try:
        from agents.setup.builder_agent.llm.api import make_client
        c = make_client(cfg.get("model"), configured=cfg.get("provider"))
        out["slots"] = f"{c.model} ({type(c).__name__.replace('Client', '')})"
    except Exception as e:  # noqa: BLE001 - shown as is
        out["slots"] = None
        out["slots_error"] = str(e)[:200]
    out["diagnosis"] = os.environ.get("BUILDER_DIAGNOSIS_MODEL") or out["slots"]
    out["escalation"] = os.environ.get("BUILDER_ESCALATION_MODEL")
    out["web_search"] = bool(os.environ.get("TAVILY_API_KEY")) and os.environ.get("CRAG_SEARCH", "cache") != "off"
    out["ollama"] = _ollama_models()
    local = out["ollama"] is not None
    sft = os.environ.get("BUILDER_DIAGNOSIS_MODEL")
    if sft and (os.environ.get("BUILDER_DIAGNOSIS_PROVIDER") or "ollama") == "ollama":
        ready = local and any(m.split(":")[0] == sft.split(":")[0] for m in out["ollama"])
        why = None if ready else (f"{sft} is not in Ollama" if local else "Ollama does not answer")
    elif out["slots"]:
        ready = local or "(Ollama)" not in out["slots"]
        why = None if ready else "Ollama does not answer"
    else:
        ready, why = False, out.get("slots_error")
    out["llm"], out["llm_missing"] = bool(ready), why       # the LLM button only when the diagnosis model answers
    return out


def _config() -> dict:
    import yaml
    for p in (ROOT / "agents.yaml", ROOT / "samples" / "agents.yaml"):
        if p.exists():
            try:
                return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("llm", {}) or {}
            except yaml.YAMLError:
                pass
    return {}


def _ollama_models() -> list[str] | None:
    import urllib.request
    from agents.setup.builder_agent.llm.ollama import _host_from_env
    try:
        with urllib.request.urlopen(_host_from_env().rstrip("/") + "/api/tags", timeout=2) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:  # noqa: BLE001 - not running: None
        return None


# ---------------------------------------------------------------- 1. the analysis (read only, instant)

def analyse() -> dict:
    """The plan of the real project: what the agent decides, would create, and reports (changes nothing)."""
    from agents.setup.builder_agent.team import TeamLayout, team_plan
    layout = TeamLayout.detect(ROOT)
    plan = team_plan(layout)
    files = [{"path": p, "mode": m} for p, m in plan.modes.items()]
    if plan.team_schema:
        files.append({"path": plan.team_schema, "mode": "team"})
    return {"code_dir": layout.code_dir, "python": plan.python_version, "train": plan.train_script,
            "serve": plan.serve_target, "target": plan.target, "separator": plan.separator,
            "features": len(plan.columns), "sample_rows": plan.sample_rows,
            "team_schema": plan.team_schema, "requirements": plan.requirements, "extra": plan.extra_packages,
            "decisions": [_decision(d) for d in plan.decisions],
            "files": files, "findings": [f.model_dump() for f in plan.findings]}


def _decision(d: dict) -> dict:
    return {"item": d.get("item"), "value": d.get("value"), "status": d.get("status"), "source": d.get("source"),
            "why": d.get("reason"), "candidates": d.get("candidates") or []}


# ---------------------------------------------------------------- 2 and 3. a run (build, or break and repair)

def start(action: str, fault: str | None, mode: str) -> Job:
    if action not in ("build", "fault"):
        raise ValueError(f"unknown action {action}")
    if action == "fault" and fault not in BY_ID:
        raise ValueError(f"unknown fault {fault}")
    if mode not in ("rules", "llm"):
        raise ValueError(f"unknown mode {mode}")
    if not _busy.acquire(blocking=False):
        raise Busy("a run is already going on")
    job = Job(action, fault if action == "fault" else None, mode)
    JOBS[job.id] = job
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return job


def _run(job: Job) -> None:
    try:
        job.result = run(job)
    except Exception as e:  # noqa: BLE001 - shown on the page
        traceback.print_exc()
        job.emit("error", text=f"{type(e).__name__}: {e}")
        job.result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            save(job)
        except OSError as e:
            print(f"[demo] recording not saved: {e}")
        _busy.release()
        job.done = True


def copy_project(dst: Path) -> Path:
    """The project as the evaluation copies it (team_eval.copy_repo), plus the root agents.yaml when there is one."""
    from agents.setup.eval import team_eval as te
    repo = te.copy_repo("wine", dst)
    if (ROOT / "agents.yaml").exists():
        shutil.copy(ROOT / "agents.yaml", repo / "agents.yaml")
    return repo


def run(job: Job) -> dict:
    from agents.setup.builder_agent.team import TeamLayout, team_plan
    st = status()
    job.emit("start", mode=job.mode, slots=st.get("slots") if job.mode == "llm" else None,
             diagnosis=st.get("diagnosis") if job.mode == "llm" else None,
             escalation=st.get("escalation") if job.mode == "llm" else None,
             web=st.get("web_search") if job.mode == "llm" else False)
    with tempfile.TemporaryDirectory(prefix="setup-demo-") as tmp:
        repo = copy_project(Path(tmp))
        job.emit("copy", text="the project is copied to a temporary folder (the real project is never changed)")
        if job.fault:
            fault = BY_ID[job.fault]
            train = team_plan(TeamLayout.detect(repo)).train_script
            where = inject(repo / train, fault, shown=train)
            job.emit("inject", label=fault.label, what=fault.what, code=fault.code.strip(), where=where,
                     expected=fault.expected)
        agent = demo_agent(job, repo)
        with listening(job, agent):
            msg = agent.step()
        ev = msg.evidence
        files = getattr(agent, "demo_files", {}) or {}
        result = {"ok": msg.kind == "pipeline.setup_completed", "kind": msg.kind, "summary": msg.summary,
                  "fixes": ev.get("fixes", []), "created": ev.get("created", []), "why": ev.get("why"),
                  "findings": ev.get("findings", []), "payload": {k: v for k, v in msg.payload.items()
                                                                 if k not in ("log_ref",)},
                  "makefile": files.get(next((p for p in files if p.endswith("Makefile")), ""), ""),
                  "minutes": round((time.time() - job.start) / 60, 1)}
        expect = BY_ID[job.fault].expect if job.fault else "build"
        # right, as the evaluation counts it: built when the error is fixable, reported (not built) when it is not
        result |= {"expect": expect, "right": result["ok"] == (expect == "build"),
                   # the error did not break this project (its versions already accept the code): nothing to show
                   "not_broken": bool(job.fault) and job.first_ok is True}
        job.emit("done", **{k: result[k] for k in ("ok", "kind", "summary", "fixes", "why", "expect", "right")})
        return result


def demo_agent(job: Job, repo: Path):
    """The normal agent on the copy, with its dry runs and rules reported. rules: no LLM at all, no memory
    (the evaluation's `rules` config); llm: the agent as configured (.env / agents.yaml), learning turned off."""
    from agents.setup.agent import SetupAgent

    class DemoAgent(SetupAgent):
        demo_files: dict = {}

        def dry_run(self, kind, inputs):
            job.builds += 1
            n = job.builds
            job.emit("build", n=n)
            r = super().dry_run(kind, inputs)
            if n == 1:
                job.first_ok = r["ok"]
            job.emit("build_done", n=n, ok=r["ok"], stage=r.get("stage"), error_class=error_name(r),
                     detail=r.get("detail"), seconds=r.get("seconds"), stages=stages(r),
                     error=error_lines(r.get("log", "")) if not r["ok"] else [])
            return r

        def fix_for(self, result, plan, env=None):
            text = SetupAgent.fix_for(result, plan, env)
            job.emit("rule", text=text, llm=self._diagnosis_client is not None)
            return text

        def act(self, decision):
            out = super().act(decision)
            self.demo_files = out.get("files") or {}
            return out

    common = dict(root=str(repo), commit=False, learn=False)
    if job.mode == "rules":
        return DemoAgent(client=None, diagnosis_client=None, escalation_client=None, crag=False, memory=None,
                         **common)
    return DemoAgent(**common)


class listening:
    """Report the decide step and each diagnosis while a run goes on (module functions the agent looks up when it
    runs; put back afterwards). One run at a time, so this is safe."""

    def __init__(self, job: Job, agent):
        self.job, self.agent = job, agent

    def __enter__(self):
        import agents.setup.builder_agent.team as team
        import agents.setup.builder_agent.team.fix as fix
        self.team, self.fix = team, fix
        self.resolve, self.diagnose = team.resolve_with_llm, fix.diagnose_team
        job, agent = self.job, self.agent

        def resolve(plan, client):
            new, log = self.resolve(plan, client)
            job.emit("plan", decisions=[_decision(d) for d in new.decisions], llm_calls=len(log or []),
                     model=getattr(client, "last_model", None) or getattr(client, "model", None) if log else None,
                     files=[p for p, m in new.modes.items() if m == "create"],
                     findings=[f.model_dump() for f in new.findings])
            return new, log

        def diagnose(plan, result, client, *a, **k):
            second = agent._escalation_client not in ("auto", None) and client is agent._escalation_client
            job.emit("ask", model=getattr(client, "model", None), escalated=second)
            d, f = self.diagnose(plan, result, client, *a, **k)
            crag = d.crag or {}
            job.emit("diagnosis", model=d.model, escalated=second, text=d.text, reason=d.reason,
                     give_up=d.give_up, valid=d.fix is not None, calls=d.calls,
                     rejected=[r for rs in d.rejected for r in rs][:6],
                     crag=crag.get("action"), query=crag.get("query"),
                     kept=[g["id"] for g in crag.get("graded", []) if g.get("kept")],
                     graded=[{"id": g["id"], "score": round(g["score"], 2), "kept": g["kept"]}
                             for g in crag.get("graded", [])],
                     web=crag.get("web_urls", [])[:3], web_source=crag.get("web_source"),
                     retrieved=d.retrieved)
            return d, f

        team.resolve_with_llm, fix.diagnose_team = resolve, diagnose
        return self

    def __exit__(self, *exc):
        self.team.resolve_with_llm, self.fix.diagnose_team = self.resolve, self.diagnose
        return False


def stages(result: dict) -> list[dict]:
    """Each make stage: passed, failed, or not reached (make stops at the first failing stage)."""
    if result.get("ok"):
        return [{"stage": s, "label": STAGE_LABEL[s], "state": "ok"} for s in ORDER]
    failed = result.get("stage")
    if failed not in ORDER:
        return [{"stage": s, "label": STAGE_LABEL[s], "state": "unknown"} for s in ORDER]
    i = ORDER.index(failed)
    return [{"stage": s, "label": STAGE_LABEL[s], "state": "ok" if j < i else "failed" if j == i else "skipped"}
            for j, s in enumerate(ORDER)]


ERROR_LINE = re.compile(r"^\s*([A-Za-z_][\w.]*(Error|Exception|Exit)|ERROR:|error:)")


NOISE = ("Collecting ", "  Using cached", "Requirement already", "make: ", "make[")


def short(line: str) -> str:
    """Paths made readable: the copy's temporary folder and the venv's site-packages are left out."""
    line = re.sub(r'"[^"]*/repo/', '"', line)
    line = re.sub(r"[^\s\"'(]*/site-packages/", ".../", line)
    return line[:220]


def error_lines(log: str, before: int = 4) -> list[str]:
    """The error itself: the last line that names an error, with the few lines above it (pip and make noise out)."""
    lines = [line for line in log.splitlines() if line.strip() and not line.startswith(NOISE)]
    hits = [i for i, line in enumerate(lines) if ERROR_LINE.match(line)]
    if not hits:
        return [short(line) for line in lines[-6:]]
    i = hits[-1]
    return [short(line) for line in lines[max(0, i - before): i + 1]]


def error_name(result: dict) -> str | None:
    """The classifier's error class; its generic "Error" replaced by the exception the last error line names."""
    name = result.get("error_class")
    if name in (None, "Error"):
        found = re.findall(r"^\s*([A-Za-z_][\w.]*(?:Error|Exception))\b", result.get("log", ""), re.M)
        return found[-1].split(".")[-1] if found else name
    return name


# ---------------------------------------------------------------- recordings (replay if the live run fails)

def _name(action: str, fault: str | None, mode: str) -> str:
    return f"{action}-{fault or 'clean'}-{mode}.json"


def save(job: Job) -> None:
    if not job.result or job.result.get("error"):
        return                                 # a crash is not worth replaying
    RECORDINGS.mkdir(exist_ok=True)
    rec = {"saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "action": job.action,
           "fault": job.fault, "mode": job.mode, "events": job.events, "result": job.result}
    (RECORDINGS / _name(job.action, job.fault, job.mode)).write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                                                     encoding="utf-8")


def recordings() -> list[dict]:
    if not RECORDINGS.exists():
        return []
    out = []
    for p in sorted(RECORDINGS.glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            out.append({"action": r["action"], "fault": r["fault"], "mode": r["mode"], "saved_at": r["saved_at"],
                        "ok": r["result"].get("ok")})
        except (ValueError, KeyError):
            continue
    return out


def recording(action: str, fault: str | None, mode: str) -> dict | None:
    p = RECORDINGS / _name(action, fault or None, mode)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------------------------------------------------------------- 4. the evaluation results (recorded files)

RESULTS = ROOT / "agents" / "setup" / "eval" / "results"


def results() -> dict:
    """The recorded evaluation, as the result files say it (markdown sections, rendered by the page)."""
    out = {}
    intervals = RESULTS / "team_intervals.md"
    if intervals.exists():
        text = intervals.read_text(encoding="utf-8")
        for name in ("team_eval_final", "team_eval_cascade"):
            m = re.search(rf"^## {name}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
            if m:
                out[name] = m.group(1).strip()
    rag = RESULTS / "rag_metrics.md"
    if rag.exists():
        out["rag"] = rag.read_text(encoding="utf-8").split("## Every diagnosis")[0].split("\n", 1)[1].strip()
    return out
