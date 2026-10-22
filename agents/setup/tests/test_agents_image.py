"""The shared agents image and compose file (roadmap 4.1, P1's shared piece; 5.1 rule 8; 5.3)."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
AGENTS = ["setup", "reviewer", "checker", "trainer", "deployer", "monitor"]


def test_compose_runs_every_agent_from_the_one_image():
    c = yaml.safe_load((ROOT / "docker-compose.agents.yml").read_text())
    assert set(c["services"]) == set(AGENTS)
    own = {"trainer": "agents/trainer/Dockerfile"}       # an agent with its own image, as its owner wrote it
    for name, dockerfile in own.items():
        s = c["services"][name]
        assert s["build"] == {"context": ".", "dockerfile": dockerfile}
        assert s["env_file"] == [{"path": ".env", "required": False}] and "./runs:/app/runs" in s["volumes"]
    assert c["networks"]["example-network"] == {"external": True}            # the baseline stack's network
    for name, s in c["services"].items():
        if name in own:
            continue
        assert s["command"] == ["python", "-m", f"agents.{name}"]
        assert s["image"] == "wine-agents:latest" and s["build"]["dockerfile"] == "Dockerfile.agents"
        assert ".:/app" in s["volumes"]                                       # agents read and write the repo
        assert s["env_file"] == [{"path": ".env", "required": False}]           # keys come at run time
        assert "host.docker.internal:host-gateway" in s["extra_hosts"]          # Ollama and the stack on the host


def test_secrets_never_enter_the_image():
    ignored = (ROOT / ".dockerignore").read_text().split()
    assert {".env", "**/.env", ".git"} <= set(ignored)
    assert "data/" not in ignored                        # the Trainer's Dockerfile copies data/ into its image


def test_image_is_python_310_with_make_and_git():
    d = (ROOT / "Dockerfile.agents").read_text()
    assert "FROM python:3.10-slim" in d and re.search(r"install -y --no-install-recommends make git", d)
    assert "pip install --no-cache-dir -r requirements.txt" in d


def test_requirements_are_pinned_and_mlflow_matches_the_server():
    reqs = [line for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    assert all("==" in r for r in reqs), [r for r in reqs if "==" not in r]
    pins = dict(r.split("==") for r in reqs)
    server = dict(r.split("==") for r in (ROOT / "baseline/mlflow/requirements.txt").read_text().split() if "==" in r)
    assert pins["mlflow"] == server["mlflow"] == "2.14.1"
    # the agents' own packages (the team file may add more, and spell names its own way: SQLAlchemy / sqlalchemy)
    norm = lambda n: re.sub(r"[-_.]+", "-", n.strip()).lower()  # noqa: E731
    roadmap = {"pandas", "numpy", "scipy", "scikit-learn", "mlflow", "pydantic", "pyyaml", "pandera", "jinja2",
               "gitpython", "httpx", "pytest"}
    assert roadmap <= {norm(n) for n in pins}, roadmap - {norm(n) for n in pins}


@pytest.mark.parametrize("value, expected", [
    ("", "http://127.0.0.1:11434"),
    ("0.0.0.0:11434", "http://127.0.0.1:11434"),                 # a server bind address, not one to call
    ("http://host.docker.internal:11434", "http://host.docker.internal:11434"),
    ("myhost:11434", "http://myhost:11434"),
])
def test_ollama_host_from_the_environment(monkeypatch, value, expected):
    from agents.setup.builder_agent.llm.ollama import _host_from_env
    monkeypatch.setenv("OLLAMA_HOST", value)
    assert _host_from_env() == expected


def test_python_m_agents_setup_plan_changes_nothing():
    before = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
    r = subprocess.run(["python3", "-m", "agents.setup", "--plan"], cwd=ROOT, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert "baseline/Makefile" in r.stdout
    assert subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout == before


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed here")
def test_docker_compose_config_parses():
    r = subprocess.run(["docker", "compose", "-f", "docker-compose.agents.yml", "config", "-q"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
