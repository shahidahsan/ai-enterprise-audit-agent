"""Shared fixtures for the test suite."""
import pytest

from src.config import get_settings

# All env vars Settings reads. deepeval's pytest plugin auto-loads the developer's
# real .env into os.environ at session start, so tests must explicitly clear these
# to stay isolated from whatever happens to be in that local .env file.
_SETTINGS_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LLM_PROVIDER",
    "AGENT_MODEL",
    "EMBEDDING_MODEL",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_PROJECT",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RETRIEVAL_TOP_K",
]


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch):
    """Reset Settings-related env vars and provide the one required key by default."""
    for var in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
