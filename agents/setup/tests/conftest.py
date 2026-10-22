import os
import tempfile

# before any agents.setup.builder_agent import: the attempt and call logs of the tests go to a temp folder, never to
# the real logs/ (a test run appended fake-model entries there, in a OneDrive-synced folder)
os.environ["BUILDER_LOG_DIR"] = tempfile.mkdtemp(prefix="builder-test-logs-")
os.environ["BUILDER_MEMORY_DIR"] = tempfile.mkdtemp(prefix="builder-test-memory-")   # learned fix memory

import pytest  # noqa: E402

from agents.setup.builder_agent.scan import rows  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _session_row_cache(tmp_path_factory):
    """One row-sample cache per test session: taxi data is hashed once, nothing piles up between runs."""
    mp = pytest.MonkeyPatch()
    mp.setattr(rows, "CACHE_FILE", tmp_path_factory.mktemp("row-cache") / "row_samples.json")
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _pypi_offline(monkeypatch):
    """No test asks PyPI: the version check is skipped (as when offline) unless a test sets its own releases."""
    from agents.setup.builder_agent.team import fix as team_fix
    monkeypatch.setattr(team_fix, "pypi_releases", lambda name, timeout=10, python=None: None)
