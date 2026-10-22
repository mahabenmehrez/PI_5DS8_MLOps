"""
agents/setup/agent.py: the Setup Agent (P1), the team version of the Builder.

The 7-step loop (roadmap 3.2 / 5.8): observe -> analyze -> decide -> dry_run -> act -> verify -> remember.
Runs once at setup, then on request.rebuild (roadmap 7).

act() is the Builder in team mode: scan the project, generate what is missing (Makefile, CI, schema, dependency
map, agents.yaml), dry-run `make -C baseline all SAMPLE=1` on a copy (dry_run_jobs/build.py), fix what the rules know how to fix (at most
max_fix_attempts), then write the files on a builder/<date> branch and commit them there for the Reviewer.
Result: pipeline.setup_completed, or pipeline.setup_failed with the stage, error class and log.
"""
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from shared.messages import Message, MessageType

TOPIC = "pipeline:setup"
HERE = Path(__file__).resolve().parent
LOGS = Path(HERE.parent.name, HERE.name, "logs", "setup")     # agents/setup/logs/setup, in the team repo (gitignored)


@dataclass
class Action:
    name: str
    level: int
    success_check: str


# Roadmap 7.3
MENU = [
    Action("write_files", 1, "every generated file parses; docker compose config passes"),
    Action("dry_run_build", 1, "make all exits 0 on a 600-row sample (roadmap: 200)"),
    Action("apply_fix", 2, "the stage that failed now passes in the dry run"),
    Action("commit_setup", 1, "commit exists (after go-live: a branch the Reviewer reviews)"),
    Action("give_up_and_report", 1, "pipeline.setup_failed posted with stage, error class and log"),
]
LEVELS = {a.name: a.level for a in MENU}


@dataclass
class Analysis:
    flag: bool
    reason: str
    trigger: Optional[Message] = None
    changed_paths: List[str] = field(default_factory=list)
    nodes: List[str] = field(default_factory=lambda: [TOPIC])


@dataclass
class Decision:
    action: str
    level: int
    params: Dict[str, Any]
    reason: str
    success_check: str


def _brief(diagnoses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A diagnosis for a board message: without the full prompts and answers (those are in logs/llm_calls.jsonl)."""
    return [{k: v for k, v in d.items() if k != "call_log"} for d in diagnoses]


def _decisions(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The decide step for a board message: each choice, its value, and who made it (rule, llm or assumed)."""
    return [{k: d[k] for k in ("item", "status", "value", "source", "reason")} for d in decisions]


class SetupAgent:
    """The Setup Agent (Owner P1). Agents never import each other: only shared/ and this folder."""

    name = "setup"
    menu = MENU

    def __init__(self, repo: str = "baseline", first_run: bool = True, root: str = ".", commit: bool = True,
                 client: Any = "auto", memory: Any = "auto", learn: bool = True, memory_dir: Optional[str] = None,
                 crag: Any = "auto", search: Any = "auto", diagnosis_client: Any = "auto",
                 escalation_client: Any = "auto"):
        self.repo = repo                    # the project's code folder (integration.layout.code_dir)
        self.root = root                    # the team repo (commands run from here)
        self.commit = commit                # False: write the files but leave the commit to a human
        self._client = client               # slots: "auto" = agents.yaml llm (Gemma on NVIDIA NIM); None: rules only
        # diagnosis: "auto" = BUILDER_DIAGNOSIS_MODEL (the fine-tuned model) when set and no slots client was
        # passed in, else the slots client; None: no LLM diagnosis
        self._diagnosis_client = diagnosis_client
        # cascade: when the diagnosis model gives up or its fix fails, this model takes the remaining attempts.
        # "auto" = BUILDER_ESCALATION_MODEL (+ BUILDER_ESCALATION_PROVIDER, default api) when set; None: no cascade
        self._escalation_client = escalation_client
        self._client_given = client != "auto"
        self._memory = memory               # "auto": the fix memory (base + learned), hybrid retrieval
        self.learn = learn                  # write verified fixes back to the memory (memory/learned/)
        self.memory_dir = Path(memory_dir) if memory_dir else None    # None: builder_agent's memory/learned/
        self.use_crag = crag                # "auto"/True: Corrective RAG over the memory; False: plain advanced RAG
        self._search = search               # "auto": Tavily (TAVILY_API_KEY), cached; None: no web search
        self.first_run = first_run
        self.inbox: List[Message] = []
        self.outbox: List[Message] = []
        self._handled: set = set()
        self.branch: Optional[str] = None             # the builder/<date> branch of the last commit
        self.history: List[Dict[str, Any]] = []      # remember: one entry per run (verified fixes only)

    def thresholds(self) -> Dict[str, Any]:
        return ((self._config().get("agents") or {}).get("setup") or {}).get("thresholds") or {}

    # 1. Look
    def observe(self) -> Dict[str, Any]:
        rebuilds = [m for m in self.inbox if m.kind == "request.rebuild" and m.id not in self._handled]
        return {"first_run": self.first_run, "rebuilds": rebuilds}

    # 2. Understand: no LLM
    def analyze(self, obs: Dict[str, Any]) -> Analysis:
        if obs["rebuilds"]:
            msg = obs["rebuilds"][0]
            return Analysis(True, f"rebuild requested: {msg.payload.get('reason', '')}", msg,
                            list(msg.payload.get("changed_paths", [])))
        if obs["first_run"]:
            return Analysis(True, "first run: the repo has no pipeline yet")
        return Analysis(False, "nothing to do")

    # 3. Decide: the fallback and the rules-only system
    def rule_decision(self, analysis: Analysis) -> Decision:
        return Decision("dry_run_build", LEVELS["dry_run_build"],
                        {"repo": self.repo, "changed_paths": analysis.changed_paths,
                         "rebuild": analysis.trigger is not None},
                        analysis.reason, MENU[1].success_check)

    def llm(self):
        """The LLM for the slots (the decide step's unsure choices), or None (no key / no server: rules only).
        The external model by API (agents.yaml llm); BUILDER_LLM_PROVIDER in .env switches it locally.
        An API model gets the local model as its fallback (API first, then Ollama, then the rules' guess, said
        as "assumed"): BUILDER_SLOTS_FALLBACK=off turns that off; BUILDER_SLOTS_TIMEOUT (s, default 60)."""
        if self._client == "auto":
            from agents.setup.builder_agent.llm.api import ApiClient, FallbackClient, load_dotenv, make_client
            cfg = self._config().get("llm", {})
            try:
                self._client = make_client(cfg.get("model"), configured=cfg.get("provider"))
            except Exception as e:  # noqa: BLE001 - no key: rules only, said in the result
                print(f"[setup] no LLM: {e}")
                self._client = None
            load_dotenv()
            if isinstance(self._client, ApiClient) and os.environ.get("BUILDER_SLOTS_FALLBACK", "on") != "off":
                self._client.timeout = float(os.environ.get("BUILDER_SLOTS_TIMEOUT", 60))
                try:
                    self._client = FallbackClient(self._client, make_client(provider="ollama"))
                except Exception as e:  # noqa: BLE001 - no local model: the API alone
                    print(f"[setup] no local fallback for the slots: {e}")
        return self._client

    def diagnosis_llm(self):
        """The LLM for the diagnosis (the fix loop), or None. BUILDER_DIAGNOSIS_MODEL (e.g. qwen-builder-sft-v5,
        the fine-tuned model) with BUILDER_DIAGNOSIS_PROVIDER (default ollama); unset: the slots LLM."""
        if self._diagnosis_client == "auto":
            from agents.setup.builder_agent.llm.api import load_dotenv, make_client
            load_dotenv()
            model = os.environ.get("BUILDER_DIAGNOSIS_MODEL")
            if self._client_given or not model:      # a client passed in (or None): one model for both
                self._diagnosis_client = self.llm()
            else:
                try:
                    client = make_client(model, provider=os.environ.get("BUILDER_DIAGNOSIS_PROVIDER") or "ollama")
                    getattr(client, "digest", None)      # Ollama: raises when the model isn't pulled
                    self._diagnosis_client = client
                except Exception as e:  # noqa: BLE001 - reported, the slots LLM takes over
                    print(f"[setup] no diagnosis LLM {model}: {e}; using the slots LLM")
                    self._diagnosis_client = self.llm()
        return self._diagnosis_client

    def escalation_llm(self):
        """The cascade's second model (an API model), or None: no cascade."""
        if self._escalation_client == "auto":
            from agents.setup.builder_agent.llm.api import load_dotenv, make_client
            load_dotenv()
            model = os.environ.get("BUILDER_ESCALATION_MODEL")
            self._escalation_client = None
            if model and not self._client_given:
                try:
                    self._escalation_client = make_client(
                        model, provider=os.environ.get("BUILDER_ESCALATION_PROVIDER") or "api")
                except Exception as e:  # noqa: BLE001 - no key: no cascade, said
                    print(f"[setup] no escalation LLM {model}: {e}")
        return self._escalation_client

    def memory(self):
        if self._memory == "auto":
            from agents.setup.builder_agent.llm.api import make_embedder
            from agents.setup.builder_agent.memory import FixMemory
            from agents.setup.builder_agent.memory.learned import memory_docs
            self._memory = FixMemory.build(make_embedder(), docs=memory_docs("learned", self.memory_dir), keywords_if_unreachable=True)
        return self._memory

    def crag(self):
        """The Corrective RAG over the current memory (rebuilt with it), or None when turned off."""
        if self.use_crag is False:
            return None
        from agents.setup.builder_agent.memory.crag import CorrectiveRAG, TavilySearch
        if self._search == "auto":
            self._search = TavilySearch()
        return CorrectiveRAG(self.memory(), search=self._search)

    def _config(self) -> Dict[str, Any]:
        for cfg in (Path(self.root) / "agents.yaml", Path(self.root) / "samples" / "agents.yaml"):
            if cfg.exists():
                try:
                    return yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
                except yaml.YAMLError:
                    pass
        return {}

    # 4. Try first: the dry run, in a subprocess with DRY_RUN=1 (roadmap 5.8)
    def dry_run(self, kind: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(inputs, f)
        try:
            r = subprocess.run([sys.executable, str(HERE.parents[1] / "dry_run_jobs" / f"{kind}.py"), f.name],
                               capture_output=True, text=True, env={**os.environ, "DRY_RUN": "1"},
                               timeout=int(inputs.get("timeout_s", 600)) + 120)
        finally:
            os.unlink(f.name)
        if r.returncode != 0 or not r.stdout.strip():
            return {"ok": False, "stage": "dry_run", "error_class": "DryRunCrashed", "detail": None,
                    "log": r.stdout + r.stderr}
        return json.loads(r.stdout.strip().splitlines()[-1])

    @staticmethod
    def fix_for(result: Dict[str, Any], plan, env: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """apply_fix (level 2): the fixes the rules know. Changes `plan` (or `env`, the dry run's settings) and
        returns what was changed; None when no rule applies (then: give_up_and_report)."""
        from agents.setup.builder_agent.scan.known import IMPORT_TO_DIST
        env = {} if env is None else env
        if result["error_class"] == "BrokenEnvironment" and not env.get("fresh_venv"):
            env["fresh_venv"] = True                # once: a second broken install is reported, not retried
            return "recreate the dry-run environment (an install was broken)"
        if result["error_class"] == "ModuleNotFoundError" and result.get("detail"):
            top = result["detail"].split(".")[0]
            root = Path(plan.layout.root)
            local = [root / top, root / f"{top}.py", root / plan.layout.code_dir / top,
                     root / plan.layout.code_dir / f"{top}.py"]
            if any(p.exists() for p in local):
                return None     # the project's own module: a path problem, never a pip install of a same-named package
            dist = IMPORT_TO_DIST.get(top, top)
            if dist not in plan.extra_packages:
                plan.extra_packages.append(dist)
                env["action"] = {"action": "add_dependency", "name": dist}
                return f"install {dist} (the code imports {top})"
        log = result.get("log", "")
        script = getattr(plan, "train_script", None)
        if result.get("stage") == "train" and script:
            from agents.setup.builder_agent.team import module_of
            root = Path(plan.layout.root)
            # change_entry_command: a relative import only works when the script runs as a module of its package
            if ("attempted relative import with no known parent package" in log and plan.train_style == "script"
                    and module_of(script, root)):
                plan.train_style = "module"
                env["action"] = {"action": "change_entry_command", "stage": "train", "style": "module"}
                return f"run train as a module (python -m {module_of(script, root)[1]}): it uses relative imports"
            # fix_path: the missing relative path exists from the script's folder or one of its parents (up to the
            # repo root): the script expects to run from there. The nearest folder wins.
            if result["error_class"] == "FileNotFoundError" and result.get("detail"):
                from agents.setup.builder_agent.team.fix import current_cwd
                missing, here = result["detail"], current_cwd(plan)
                if not Path(missing).is_absolute() and not (root / here / missing).exists():
                    folder = Path(script).parent
                    while True:
                        f = folder.as_posix() or "."
                        if f != here and (root / f / missing).exists():
                            plan.train_cwd = f
                            env["action"] = {"action": "fix_path", "target": "workdir", "path": f}
                            where = "next to the script" if f == (Path(script).parent.as_posix() or ".") \
                                else f"in {f}/, above the script"
                            return f"run train from {f}/ ({missing} is there, {where})"
                        if f == ".":
                            break
                        folder = folder.parent
        return None

    # 5. Do
    def act(self, decision: Decision) -> Dict[str, Any]:
        if decision.action != "dry_run_build":
            raise ValueError(f"unknown action {decision.action}")
        from agents.setup.builder_agent.team import TeamLayout, render_team, team_plan, write_team
        layout = TeamLayout.detect(self.root)     # the team layout, or a small repo's (the test repos)
        plan = team_plan(layout)
        # the decide step: choices the rules couldn't settle (needs_llm) go to the LLM, the rest stay as decided
        from agents.setup.builder_agent.team import resolve_with_llm
        needs = any(d["status"] == "needs_llm" for d in plan.decisions)
        plan, decide_log = resolve_with_llm(plan, self.llm() if needs else None)
        th = self.thresholds()
        plan.sample_rows = int(th.get("dry_run_sample_rows", plan.sample_rows))
        max_fixes, timeout_s = int(th.get("max_fix_attempts", 3)), int(th.get("dry_run_timeout_s", 600))
        from agents.setup.builder_agent.team.fix import apply, diagnose_team, regressed
        fixes: List[str] = []                  # what was applied, in order (reverted ones say so)
        attempts: List[Dict[str, Any]] = []    # one per applied fix, for the memory write-back
        diagnoses: List[Dict[str, Any]] = []   # what the LLM was asked and answered
        reverted: List[Dict[str, Any]] = []
        tried: List[Dict[str, Any]] = []       # every fix applied (rules and LLM): never proposed twice
        escalated = False                      # cascade: the escalation model has taken over
        env: Dict[str, Any] = {"fresh_venv": False}
        rebuild = decision.params.get("rebuild", False)
        from agents.setup.builder_agent.render import MARKER

        def run() -> Dict[str, Any]:
            return self.dry_run("build", {"root": self.root, "files": render_team(plan), "timeout_s": timeout_s,
                                          "makefile": layout.makefile, "marker": MARKER,
                                          "update_own": [layout.makefile, layout.ci] if rebuild else [], **env})

        def failed(result, why: str) -> Dict[str, Any]:
            return {"ok": False, "result": result, "fixes": fixes, "plan": plan, "why": why, "diagnoses": diagnoses,
                    "attempts": attempts, "decide_log": decide_log}

        result = run()
        while not result["ok"]:
            if len(fixes) >= max_fixes:
                return failed(result, f"{max_fixes} fixes tried")
            rule = self.fix_for(result, plan, env)
            if rule is not None:
                # stored in memory as the menu action it amounts to; recreating the venv is not a project fix
                action = env.pop("action", None)   # recreating the venv is not a project fix: not remembered
                if action is not None:
                    tried.append(action)
                trial, source, text, reason, retrieval = run(), "rule", rule, "rule", None
            else:
                second = self.escalation_llm()
                client = second if escalated else self.diagnosis_llm()
                if client is None:
                    return failed(result, "no rule fixes this error and no LLM is configured")
                crag = self.crag()

                def ask(model):
                    return diagnose_team(plan, result, model, fixes, len(fixes) + 1, max_fixes,
                                         memory=self.memory(), reverted=reverted, crag=crag,
                                         scan_summary=self._scan_summary(plan), tried=tried)
                d, fix = ask(client)
                diagnoses.append(d.model_dump())
                if fix is None and second is not None and not escalated:
                    # cascade: the first model gave up or gave no valid answer: the second one, same attempt
                    escalated = True
                    d, fix = ask(second)
                    diagnoses.append(d.model_dump() | {"escalated": True})
                elif escalated:
                    diagnoses[-1]["escalated"] = True
                if fix is None:
                    return failed(result, f"LLM: {d.text}" + (f" ({d.reason})" if d.reason else ""))
                before = (list(plan.extra_packages), plan.train_cwd, plan.train_style)
                tried.append(fix.model_dump())
                apply(plan, fix)
                trial, source, text, action, reason = run(), "llm", d.text, d.fix, d.reason
                retrieval = d.crag or {"action": "off", "memory_cases": d.retrieved}
                stuck = (trial["stage"], trial.get("error_class")) == (result["stage"], result.get("error_class"))
                if not trial["ok"] and second is not None and (stuck or regressed(result, trial)):
                    escalated = True                 # cascade: its fix changed nothing or broke more: the second model goes on
                if regressed(result, trial):
                    plan.extra_packages[:], plan.train_cwd, plan.train_style = before
                    reverted.append(d.fix)
                    fixes.append(f"{text}: reverted (it broke the earlier stage {trial['stage']})")
                    attempts.append(self._attempt(len(fixes), result, action, text, reason, source,
                                                  f"reverted: regression (failed at {trial['stage']})", retrieval))
                    continue
            fixes.append(text)
            if action is not None:
                attempts.append(self._attempt(len(fixes), result, action, text, reason, source,
                                              "passed" if trial["ok"] else f"failed at {trial['stage']}", retrieval))
            result = trial
        files = render_team(plan)
        from agents.setup.builder_agent.repo_writer import current_branch, is_git_repo
        start_branch = current_branch(Path(self.root)) if is_git_repo(Path(self.root)) else None
        report = write_team(plan, files, self.root, update_own=rebuild)
        sha = self._commit(report.created + report.updated, decision.reason, start_branch) if self.commit else None
        return {"ok": True, "result": result, "fixes": fixes, "plan": plan, "report": report, "commit": sha,
                "files": files, "diagnoses": diagnoses, "attempts": attempts, "decide_log": decide_log}

    @staticmethod
    def _attempt(n, result, action, text, reason, source, outcome, retrieval=None) -> Dict[str, Any]:
        """retrieval: what the LLM was shown (CRAG action, kept memory cases, web query and URLs), so a learned
        fix says where its knowledge came from; None for a rule fix."""
        return {"n": n, "stage": result["stage"], "error_tail": result.get("log", "").splitlines()[-50:],
                "action": action, "action_text": text, "reason": reason, "source": source, "outcome": outcome,
                "retrieval": retrieval,
                "sandbox_attempt_id": f"team-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}-{n}"}

    @staticmethod
    def _scan_summary(plan) -> str:
        return "\n".join([f"code: {plan.layout.code_dir}/, Python {plan.python_version}",
                          f"train: {plan.train_script}, serve: {plan.serve_target}",
                          f"data: {plan.layout.data_path} (separator {plan.separator!r}, target {plan.target})",
                          *(f"finding: {f.message}" for f in plan.findings)])

    def _commit(self, created: List[str], reason: str, start_branch: Optional[str]) -> Optional[str]:
        """Commit the created files on builder/<date>, then go back to the branch the repo was on: the commit
        waits on its branch for the Reviewer (roadmap 7: after go-live, a branch the Reviewer reviews)."""
        from agents.setup.builder_agent.decide.contracts import load_contracts
        from agents.setup.builder_agent.repo_writer import RepoWriter, current_branch, is_git_repo
        root = Path(self.root).resolve()
        if not created or not is_git_repo(root):
            return None
        writer = RepoWriter(root)                    # already on builder/<date>: write_team switched there
        writer.created = set(created)
        author = load_contracts(HERE / "contracts.yaml").i("reviewer", "builder_commit_author")
        branch = current_branch(root)
        sha = writer.commit(f"Setup Agent: pipeline files ({reason})"[:72] + f"\n\nOn {branch}.", author)
        if start_branch and start_branch != branch:
            writer.restore(start_branch)
        self.branch = branch
        return sha

    # 6. Check, then post the result
    def verify(self, decision: Decision, out: Dict[str, Any]) -> Message:
        plan = out["plan"]
        lay = plan.layout
        if out["ok"]:
            payload = {"repo": lay.code_dir, "commit": out["commit"] or "", "files": {
                "Makefile": lay.makefile, "ci": lay.ci, "schema": plan.team_schema or lay.schema_file,
                "agents_yaml": lay.agents_yaml,
                "dependency_map": lay.depmap}, "agents_yaml": lay.agents_yaml,
                "schema_path": plan.team_schema or lay.schema_file,
                "depmap_seeded": True, "branch": self.branch}
            created = out["report"].created
            msg = Message(type=MessageType.FINDING, kind="pipeline.setup_completed", sender=self.name, to=[],
                          topic=TOPIC, level=1, confidence=1.0,
                          summary=(f"Pipeline ready for {lay.code_dir}: make all passes on a sample, "
                                   f"{len(created)} files created, {len(out['fixes'])} fixes")[:160],
                          evidence={"dry_run_seconds": out["result"].get("seconds"), "fix_attempts": len(out["fixes"]),
                                    "fixes": out["fixes"], "diagnoses": _brief(out.get("diagnoses", [])), "created": created,
                                    "updated": out["report"].updated,
                                    "decisions": _decisions(plan.decisions), "decide_calls": len(out.get("decide_log") or []),
                                    "verified": out["report"].verified,
                                    "findings": [f.model_dump() for f in plan.findings]},
                          payload=payload)
        else:
            r = out["result"]
            logs = Path(self.root) / LOGS
            logs.mkdir(parents=True, exist_ok=True)
            log_ref = (LOGS / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{r['stage']}.log").as_posix()
            (Path(self.root) / log_ref).write_text(r.get("log", ""), encoding="utf-8")
            msg = Message(type=MessageType.FINDING, kind="pipeline.setup_failed", sender=self.name, to=["human"],
                          topic=TOPIC, level=1, confidence=1.0,
                          summary=(f"Setup failed at {r['stage']} ({r['error_class']}) after "
                                   f"{len(out['fixes'])} fixes")[:160],
                          evidence={"detail": r.get("detail"), "fixes": out["fixes"], "why": out.get("why"),
                                    "decisions": _decisions(plan.decisions), "decide_calls": len(out.get("decide_log") or []),
                                    "diagnoses": _brief(out.get("diagnoses", []))},
                          payload={"stage": r["stage"], "error_class": r["error_class"],
                                   "attempts": len(out["fixes"]),
                                   "log_ref": log_ref})
        self.outbox.append(msg)
        return msg

    # 7. Remember: every fix the dry run checked goes to memory/learned/ with its outcome; retrieval only
    # uses the ones that passed (builder_agent.memory.learned: "learned" = base + passed fixes)
    def remember(self, out: Dict[str, Any]) -> None:
        verified = [a["action_text"] for a in out.get("attempts", []) if a["outcome"] == "passed"]
        self.history.append({"ok": out["ok"], "fixes": out["fixes"] if out["ok"] else [], "verified": verified,
                             "error_class": None if out["ok"] else out["result"]["error_class"]})
        if not (self.learn and out.get("attempts")):
            return
        from types import SimpleNamespace

        from agents.setup.builder_agent.memory.learned import write_back
        client = self._diagnosis_client if self._diagnosis_client not in ("auto", None) else None
        from agents.setup.builder_agent.team.fix import PROMPT
        crag_on = self.use_crag is not False
        run = SimpleNamespace(repo=out["plan"].layout.code_dir, model=getattr(client, "model", None),
                              model_digest=getattr(client, "digest", None),
                              prompt_version=PROMPT,
                              retriever="crag" if crag_on else "advanced",
                              attempts=[SimpleNamespace(**a) for a in out["attempts"]])
        write_back(run, memory_set="learned", directory=self.memory_dir)
        if self._memory not in ("auto", None):
            self._memory = "auto"                    # rebuilt on next use, with what was just learned

    def _log(self, decision: "Decision", out: Dict[str, Any], msg: Message) -> None:
        """Every LLM call, action, check and outcome is logged (roadmap 13.1.6): agents/setup/logs/runs.jsonl
        (one line per step) and llm_calls.jsonl (one line per call, prompts and answers in full)."""
        logs = Path(self.root) / LOGS.parent
        logs.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        run = {"time": now, "message_id": msg.id, "kind": msg.kind, "decision": decision.action,
               "reason": decision.reason, "level": decision.level, "fixes": out.get("fixes", []),
               "attempts": [{k: a[k] for k in ("n", "stage", "action_text", "source", "outcome")}
                            for a in out.get("attempts", [])],
               "checks": out.get("checks", {}), "dry_run_seconds": out["result"].get("seconds"),
               "why": out.get("why"), "decisions": _decisions(out["plan"].decisions)}
        with open(logs / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(run, ensure_ascii=False) + "\n")
        with open(logs / "llm_calls.jsonl", "a", encoding="utf-8") as f:
            for c in out.get("decide_log", []):
                f.write(json.dumps({"time": now, "message_id": msg.id, "step": "decide", **c},
                                   ensure_ascii=False) + "\n")
            for n, d in enumerate(out.get("diagnoses", []), 1):
                for c in d.get("call_log", []):
                    f.write(json.dumps({"time": now, "message_id": msg.id, "diagnosis": n, "model": d.get("model"),
                                        "crag_action": (d.get("crag") or {}).get("action"), **c},
                                       ensure_ascii=False) + "\n")

    def compose_check(self) -> str:
        """write_files' success check (roadmap 7.3): docker compose config passes on the stack's compose file."""
        import shutil
        files = [p for p in (Path(self.root) / self.repo / "docker-compose.yaml",
                             Path(self.root) / self.repo / "docker-compose.yml") if p.is_file()]
        if not files:
            return "no compose file"
        if shutil.which("docker") is None:
            return "skipped (docker not installed here)"
        r = subprocess.run(["docker", "compose", "-f", str(files[0]), "config", "-q"], capture_output=True,
                           text=True, timeout=60)
        return "passes" if r.returncode == 0 else f"fails: {r.stderr.strip()[:300]}"

    def step(self) -> Optional[Message]:
        """One turn of the loop. Returns the result message, or None when there is nothing to do."""
        analysis = self.analyze(self.observe())
        if not analysis.flag:
            return None
        decision = self.rule_decision(analysis)
        out = self.act(decision)
        out["checks"] = {"docker_compose_config": self.compose_check()}
        msg = self.verify(decision, out)
        msg.evidence["checks"] = out["checks"]
        self.remember(out)
        self._log(decision, out, msg)
        if analysis.trigger is not None:
            msg.reply_to = analysis.trigger.id
            msg.incident_id = analysis.trigger.incident_id
            self.mark_handled(analysis.trigger)
        self.first_run = False
        return msg

    # Messages from others
    def handle(self, msg: Message) -> Optional[Message]:
        """request.rebuild -> "accepted" answer (the work happens in the next step()). Idempotent."""
        if msg.id in self._handled:
            return None
        if msg.kind != "request.rebuild":
            print(f"[setup] ignoring unknown kind {msg.kind}")
            return None
        self.inbox.append(msg)
        answer = Message(
            type=MessageType.ANSWER, kind="request.rebuild", sender=self.name, to=[msg.sender],
            topic=TOPIC, reply_to=msg.id, incident_id=msg.incident_id, level=1,
            summary=f"Rebuild accepted: {msg.payload.get('reason', '')}"[:160],
            payload={"status": "accepted", "changed_paths": msg.payload.get("changed_paths", [])},
        )
        self.outbox.append(answer)
        return answer

    def mark_handled(self, msg: Message) -> None:
        self._handled.add(msg.id)
