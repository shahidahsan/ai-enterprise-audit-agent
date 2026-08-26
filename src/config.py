"""Application configuration loaded from environment variables / .env."""
from functools import lru_cache

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str
    openai_api_key: str
    agent_model: str = "claude-sonnet-5"
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

    @computed_field
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
