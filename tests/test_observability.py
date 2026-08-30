"""Tests for the LangSmith tracing configuration bridge."""
import os

from src.config import Settings
from src.observability.tracing import configure_langsmith


def test_configure_langsmith_sets_tracing_env_vars():
    settings = Settings(
        _env_file=None,
        openai_api_key="k",
        langchain_api_key="ls-test-key",
        langchain_project="my-project",
        langchain_tracing_v2=True,
    )

    configure_langsmith(settings)

    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_PROJECT"] == "my-project"
    assert os.environ["LANGCHAIN_API_KEY"] == "ls-test-key"


def test_configure_langsmith_disables_tracing_when_settings_says_false():
    settings = Settings(_env_file=None, openai_api_key="k", langchain_tracing_v2=False)

    configure_langsmith(settings)

    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_configure_langsmith_skips_api_key_when_not_configured(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    settings = Settings(_env_file=None, openai_api_key="k", langchain_api_key=None)

    configure_langsmith(settings)

    assert "LANGCHAIN_API_KEY" not in os.environ
