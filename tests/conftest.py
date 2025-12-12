import os

import pytest


@pytest.fixture(autouse=True)
def stub_env(monkeypatch):
    """Keep required config available without needing real secrets."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", "test-key"))
    monkeypatch.setenv("DEFAULT_MODEL", os.getenv("DEFAULT_MODEL", "claude-3-opus"))
    monkeypatch.setenv("MCP_SERVER_URL", os.getenv("MCP_SERVER_URL", "http://localhost:9000"))
    yield
