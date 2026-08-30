"""Bridges Settings into the env vars LangChain's global tracer reads from."""
import os

from src.config import Settings


def configure_langsmith(settings: Settings) -> None:
    """Push LangSmith tracing config into os.environ so LangChain auto-traces calls."""
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if settings.langchain_tracing_v2 else "false"
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
