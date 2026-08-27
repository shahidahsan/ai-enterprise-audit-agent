"""Tests for application settings loaded from environment variables."""
import pytest
from pydantic import ValidationError

from src.config import Settings


def test_settings_loads_openai_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key == "test-openai-key"
    assert settings.anthropic_api_key is None


def test_settings_applies_sensible_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    settings = Settings(_env_file=None)

    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.llm_provider == "openai"
    assert settings.agent_model == "gpt-4o-mini"
    assert settings.langchain_project == "compliance-audit-agent"
    assert settings.langchain_tracing_v2 is True
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 150
    assert settings.retrieval_top_k == 5


def test_settings_missing_openai_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_rejects_unknown_llm_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_switching_to_anthropic_requires_anthropic_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_can_switch_llm_provider_to_anthropic(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("AGENT_MODEL", "claude-sonnet-5")

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "anthropic"
    assert settings.agent_model == "claude-sonnet-5"


def test_settings_builds_postgres_dsn_from_parts(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "audit")
    monkeypatch.setenv("POSTGRES_USER", "auditor")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")

    settings = Settings(_env_file=None)

    assert settings.postgres_dsn == "postgresql+psycopg://auditor:secret@db.internal:5433/audit"
