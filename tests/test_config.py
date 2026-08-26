"""Tests for application settings loaded from environment variables."""
import pytest
from pydantic import ValidationError

from src.config import Settings


def test_settings_loads_required_keys_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    settings = Settings(_env_file=None)

    assert settings.anthropic_api_key == "test-anthropic-key"
    assert settings.openai_api_key == "test-openai-key"


def test_settings_applies_sensible_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    settings = Settings(_env_file=None)

    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.langchain_project == "compliance-audit-agent"
    assert settings.langchain_tracing_v2 is True
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 150
    assert settings.retrieval_top_k == 5


def test_settings_missing_required_keys_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_builds_postgres_dsn_from_parts(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "audit")
    monkeypatch.setenv("POSTGRES_USER", "auditor")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")

    settings = Settings(_env_file=None)

    assert settings.postgres_dsn == "postgresql://auditor:secret@db.internal:5433/audit"
