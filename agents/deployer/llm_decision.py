"""
Decision par LLM du Deployer (zone grise uniquement).

Principes (roadmap 5.9) :
  - temperature 0 et sortie JSON ;
  - la reponse est VALIDEE : action connue, autorisee, bon niveau, params valides ;
  - une erreur -> UN seul nouvel essai avec le message d'erreur ajoute ;
  - un deuxieme echec -> LLMInvalid : l'agent utilise alors ses regles ;
  - le LLM ne peut proposer que `hold` ou `rollback` (jamais avancer, jamais promouvoir) ;
  - chaque appel est journalise (prompt, reponse brute, decision, jetons, latence).

Configuration :  DEPLOYER_LLM_MODEL, OLLAMA_URL, ANTHROPIC_API_KEY, DEPLOYER_LLM_LOG
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Tuple

PROMPT_PATH = Path(__file__).with_name("prompt.md")
ALLOWED = ("hold", "rollback")          # le LLM ne peut jamais avancer ni promouvoir
DEFAULT_MODELS = {"ollama": "llama3.2:3b", "api": "claude-haiku-5-5"}
MAX_USER_CHARS = 8000                    # ~2000 jetons : jamais de donnees brutes


class LLMInvalid(Exception):
    """Reponse du LLM inutilisable apres un nouvel essai."""


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_user_message(context: dict, menu) -> str:
    allowed = [{"action": a.name, "level": a.level, "params": a.params_schema,
                "success_check": a.success_check} for a in menu if a.name in ALLOWED]
    text = json.dumps({"situation": context, "allowed_actions": allowed},
                      ensure_ascii=False, default=str)
    return text[:MAX_USER_CHARS]


def extract_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMInvalid("aucun objet JSON dans la reponse")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMInvalid(f"JSON invalide : {exc}")


def validate(parsed: dict, menu) -> dict:
    levels = {a.name: a.level for a in menu}
    action = parsed.get("action")
    if action not in levels:
        raise LLMInvalid(f"action inconnue : {action!r}")
    if action not in ALLOWED:
        raise LLMInvalid(f"action interdite pour le LLM : {action!r} (autorisees : {list(ALLOWED)})")
    if parsed.get("level") != levels[action]:
        raise LLMInvalid(f"niveau faux pour {action} : attendu {levels[action]}, recu {parsed.get('level')!r}")
    params = parsed.get("params")
    if not isinstance(params, dict) or not str(params.get("reason", "")).strip():
        raise LLMInvalid("params doit etre un objet avec une cle 'reason' non vide")
    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise LLMInvalid("'reason' doit etre une phrase non vide")
    conf = parsed.get("confidence")
    if conf is not None and not (isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0):
        raise LLMInvalid("'confidence' doit etre un nombre entre 0 et 1")
    check = parsed.get("success_check")
    if not isinstance(check, str) or not check.strip():
        raise LLMInvalid("'success_check' doit etre une phrase non vide")
    return {"action": action, "level": parsed["level"], "params": params,
            "reason": reason, "confidence": conf, "success_check": check}


def _post(url: str, body: dict, headers: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_transport(model: str, base_url: str) -> Callable[[str, str], Tuple[str, dict]]:
    def call(system: str, user: str):
        data = _post(f"{base_url}/api/chat", {
            "model": model, "stream": False, "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}, {})
        tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        return data["message"]["content"], {"tokens": tokens}
    return call


def anthropic_transport(model: str) -> Callable[[str, str], Tuple[str, dict]]:
    def call(system: str, user: str):
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY manquante")
        data = _post("https://api.anthropic.com/v1/messages", {
            "model": model, "max_tokens": 400, "temperature": 0,
            "system": system, "messages": [{"role": "user", "content": user}]},
            {"x-api-key": key, "anthropic-version": "2023-06-01"})
        text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        usage = data.get("usage", {})
        return text, {"tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)}
    return call


class DeployerLLM:
    """Appelable par DeployerAgent : llm(context, menu) -> decision validee."""

    def __init__(self, transport: Callable[[str, str], Tuple[str, dict]],
                 provider: str = "custom", model: str = "custom",
                 log_path: Optional[str] = None):
        self.transport, self.provider, self.model = transport, provider, model
        self.log_path = log_path or os.getenv("DEPLOYER_LLM_LOG")
        self.calls: list[dict] = []

    def _record(self, entry: dict) -> None:
        self.calls.append(entry)
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def __call__(self, context: dict, menu) -> dict:
        system = load_prompt()
        user = build_user_message(context, menu)
        error_note = ""
        for attempt in (1, 2):
            started = time.time()
            entry = {"time": datetime.now(timezone.utc).isoformat(), "provider": self.provider,
                     "model": self.model, "attempt": attempt, "prompt_user": user,
                     "retry_note": error_note, "raw": None, "parsed": None,
                     "valid": False, "error": None, "tokens": 0, "latency_s": 0.0}
            try:
                raw, meta = self.transport(
                    system, user + (f"\n\nTa reponse precedente etait invalide : {error_note}. "
                                    "Reponds uniquement par un objet JSON valide." if error_note else ""))
                entry.update(raw=raw, tokens=meta.get("tokens", 0))
                decision = validate(extract_json(raw), menu)
                entry.update(parsed=decision, valid=True)
                entry["latency_s"] = round(time.time() - started, 3)
                self._record(entry)
                return decision
            except LLMInvalid as exc:
                error_note = str(exc)
                entry.update(error=error_note, latency_s=round(time.time() - started, 3))
                self._record(entry)
            except Exception as exc:                  # reseau, cle manquante, modele absent...
                entry.update(error=f"transport: {exc}", latency_s=round(time.time() - started, 3))
                self._record(entry)
                raise
        raise LLMInvalid(f"deux reponses invalides de suite : {error_note}")


def make_llm(provider: str = "off", model: Optional[str] = None,
             log_path: Optional[str] = None) -> Optional[DeployerLLM]:
    """provider : 'off' (regles seulement), 'ollama' (local) ou 'api' (Claude)."""
    provider = (provider or "off").lower()
    if provider == "off":
        return None
    model = model or os.getenv("DEPLOYER_LLM_MODEL") or DEFAULT_MODELS[provider]
    if provider == "ollama":
        transport = ollama_transport(model, os.getenv("OLLAMA_URL", "http://localhost:11434"))
    elif provider == "api":
        transport = anthropic_transport(model)
    else:
        raise ValueError(f"fournisseur inconnu : {provider}")
    return DeployerLLM(transport, provider, model, log_path)
