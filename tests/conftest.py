"""Shared fixtures for the test suite."""
from unittest.mock import MagicMock

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


@pytest.fixture
def make_mock_llm():
    """Factory for a fake chat model standing in for a real LangChain BaseChatModel.

    ``tool_call_responses`` is the scripted sequence of AIMessages returned by
    successive calls to ``llm.bind_tools(tools).invoke(...)``. ``structured_result``
    is what ``llm.with_structured_output(Schema).invoke(...)`` returns.
    """

    def _factory(tool_call_responses, structured_result):
        llm = MagicMock(name="mock_chat_model")
        llm.bind_tools.return_value.invoke.side_effect = tool_call_responses
        llm.with_structured_output.return_value.invoke.return_value = structured_result
        return llm

    return _factory
