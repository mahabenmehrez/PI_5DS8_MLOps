"""The decide step of team mode: each choice the plan needs is "decided" by a rule when the evidence is clear,
or "needs_llm" when it isn't, with the reason and the candidates (the Builder's design, decide/models.py).
Only the needs_llm choices go to the LLM, which must answer with one of the candidates (prompts/team_slots_v1.md,
strict JSON, validated, one retry with the reasons). Without an LLM, or without a valid answer, the rule's best
guess stays and is reported as an assumption: never a silent guess.

Choices: python (the Builder's choose_python), target (the column the model predicts), train_script.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel, Field

from ..decide.contracts import load_contracts
from ..decide.rules import choose_python
from ..llm.prompts import load_prompt

PROMPT = "team_slots_v1"
TEAM_PYTHON = "3.10"                       # roadmap 5.3: one image, Python 3.10 like the baseline
MAX_CALLS = 2


class Decision(BaseModel):
    item: str                              # python | target | train_script
    status: str                            # decided | needs_llm
    value: str | None                      # the rule's choice (decided) or best guess (needs_llm)
    reason: str
    candidates: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)        # what the LLM is shown about this choice
    source: str = "rule"                   # rule | llm | assumed (needs_llm, no valid LLM answer)


def decide_python(ctx, contracts_path) -> Decision:
    choice, need = choose_python(ctx, load_contracts(contracts_path))
    hints = [f"{h.value} ({h.kind}: {h.source})" for h in ctx.python_version_hints]
    if need is None:
        if choice.assumed:                      # nothing declared: the team's Python, not the Builder's default
            return Decision(item="python", status="decided", value=TEAM_PYTHON, source="assumed",
                            reason=f"no Python version declared; the team's {TEAM_PYTHON} (roadmap 5.3)")
        return Decision(item="python", status="decided", value=choice.version, reason=choice.reason)
    candidates = sorted({h.value for h in ctx.python_version_hints if h.value.count(".") == 1
                         and h.value.replace(".", "").isdigit()}) or ["3.10"]
    return Decision(item="python", status="needs_llm", value=candidates[-1], reason=need.reason,
                    candidates=candidates, facts=hints)


def decide_target(ctx, header: list[str], script: tuple[str, str] | None = None) -> Decision:
    """decided: one candidate in the data, or the strongest has at least twice the evidence of the next one.
    needs_llm: several candidates with close evidence, or none in the data (then any data column).
    script: (path, text) of the chosen training script: its lines that use each candidate are shown to the LLM."""
    found = [c for c in ctx.target_candidates if c.column in header]
    facts = [f"{c.column}: evidence score {c.score} ({'; '.join(c.evidence[:3])})" for c in ctx.target_candidates]
    facts.append("data columns: " + ", ".join(header))
    if script:
        path, text = script
        for c in found:
            uses = [line.strip() for line in text.splitlines() if f'"{c.column}"' in line or f"'{c.column}'" in line]
            facts.append(f"{c.column} in {path} (the training script): " + (" | ".join(uses[:3]) or "not used"))
    if len(found) == 1:
        return Decision(item="target", status="decided", value=found[0].column, facts=facts,
                        reason=f"the only target candidate in the data ({found[0].score} evidence)")
    if len(found) > 1 and found[0].score >= 2 * found[1].score:
        return Decision(item="target", status="decided", value=found[0].column, facts=facts,
                        reason=f"{found[0].column} has at least twice the evidence of {found[1].column} "
                               f"({found[0].score} vs {found[1].score})")
    if found:
        return Decision(item="target", status="needs_llm", value=found[0].column, facts=facts,
                        candidates=[c.column for c in found],
                        reason="several columns could be the target with close evidence: "
                               + ", ".join(f"{c.column} ({c.score})" for c in found))
    return Decision(item="target", status="needs_llm", value=header[-1], candidates=list(header), facts=facts,
                    reason="no column of the data is used as a target in the code")


def decide_train_script(paths: list[str], code: Path, saves: set[str] = frozenset()) -> Decision:
    """decided: one script trains, or several do and exactly one saves a model (the pipeline's script produces the
    model that is served). needs_llm: several, and none or several save one."""
    if len(paths) <= 1:
        return Decision(item="train_script", status="decided", value=paths[0] if paths else None,
                        reason="the only script that trains a model" if paths else "no script trains a model")
    saving = [p for p in paths if p in saves]
    if len(saving) == 1:
        return Decision(item="train_script", status="decided", value=saving[0], candidates=list(paths),
                        reason=f"{len(paths)} scripts train a model; only {saving[0]} saves one")
    facts = []
    for p in paths:
        lines = (code / p).read_text(encoding="utf-8", errors="replace").splitlines()
        facts.append(f"{p} ({'saves a model' if p in saves else 'saves no model'}): "
                     + " | ".join(line.strip() for line in lines[:12] if line.strip())[:400])
    return Decision(item="train_script", status="needs_llm", value=paths[0], candidates=list(paths), facts=facts,
                    reason=f"{len(paths)} scripts train a model: " + ", ".join(paths))


def ask_llm(decisions: list[Decision], client, project_facts: str) -> tuple[dict[str, str], list[dict]]:
    """One call for all the needs_llm choices: strict JSON, each answer one of its candidates."""
    todo = [d for d in decisions if d.status == "needs_llm"]
    schema = {"type": "object", "additionalProperties": False, "required": [d.item for d in todo],
              "properties": {d.item: {"type": "string", "enum": d.candidates} for d in todo}}
    prompt = load_prompt(PROMPT)
    variables = {"project_facts": project_facts,
                 "questions_text": "\n".join(
                     f"- {d.item}: {d.reason}\n  allowed: {', '.join(d.candidates)}\n"
                     + "".join(f"  fact: {f}\n" for f in d.facts) for d in todo)}
    messages = [{"role": "system", "content": prompt.render("system", variables)},
                {"role": "user", "content": prompt.render("user", variables)}]
    log = []
    for n in range(1, MAX_CALLS + 1):
        t0 = time.perf_counter()
        try:
            resp = client.chat(messages, schema=schema)
        except Exception as e:  # noqa: BLE001 - unreachable LLM: the rule's guesses stay (assumed)
            log.append({"messages": messages, "response": None, "valid": False, "reasons": [],
                        "error": f"{type(e).__name__}: {e}", "latency_s": round(time.perf_counter() - t0, 2)})
            return {}, log
        reasons = []
        try:
            answer = json.loads(resp.content)
        except json.JSONDecodeError as e:
            answer, reasons = {}, [f"the answer is not valid JSON: {e}"]
        if not isinstance(answer, dict):
            answer, reasons = {}, ["the answer must be one JSON object"]
        for d in todo:
            if answer.get(d.item) not in d.candidates:
                reasons.append(f"{d.item}: {answer.get(d.item)!r} is not one of {', '.join(d.candidates)}")
        log.append({"model": getattr(client, "last_model", None) or getattr(client, "model", None),
                    "messages": messages, "response": resp.content, "valid": not reasons, "reasons": reasons,
                    "latency_s": resp.latency_s, "prompt_tokens": resp.prompt_tokens,
                    "output_tokens": resp.output_tokens})
        if not reasons:
            return {d.item: answer[d.item] for d in todo}, log
        messages = messages + [{"role": "assistant", "content": resp.content},
                               {"role": "user", "content": prompt.render("retry", variables | {
                                   "reasons_text": "\n".join(f"- {r}" for r in reasons)})}]
    return {}, log
