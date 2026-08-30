"""Pydantic schema for the agent's structured final output."""
from typing import Literal

from pydantic import BaseModel, Field


class AuditFinding(BaseModel):
    """The agent's structured final answer -- the Pydantic output guardrail."""

    summary: str = Field(..., description="Concise audit finding.")
    data_points: list[str] = Field(default_factory=list, description="Key numerical values extracted.")
    compliance_status: Literal["PASSED", "FLAGGED", "NEEDS_REVIEW"]
    sources: list[str] = Field(default_factory=list, description="Document sections cited.")
