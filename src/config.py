"""Application configuration loaded from environment variables / .env."""
from functools import lru_cache
from typing import Literal

from pydantic import computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    anthropic_api_key: str | None = None
    llm_provider: Literal["openai", "anthropic"] = "openai"
    agent_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    langchain_api_key: str | None = None
    langchain_tracing_v2: bool = True
    langchain_project: str = "compliance-audit-agent"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "audit_agent"
    postgres_user: str = "audit_agent"
    postgres_password: str = "audit_agent"

    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_top_k: int = 5

    @model_validator(mode="after")
    def _require_key_for_selected_provider(self) -> "Settings":
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("anthropic_api_key is required when llm_provider is 'anthropic'")
        return self

    @computed_field
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
