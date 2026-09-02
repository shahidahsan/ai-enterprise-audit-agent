"""Pydantic schema for the agent's structured final output."""
from typing import Literal

from pydantic import BaseModel, Field


class AuditFinding(BaseModel):
    """The agent's structured final answer -- the Pydantic output guardrail."""

    summary: str = Field(..., description="Concise audit finding.")
    data_points: list[str] = Field(default_factory=list, description="Key numerical values extracted.")
    compliance_status: Literal["PASSED", "FLAGGED", "NEEDS_REVIEW"]
    sources: list[str] = Field(default_factory=list, description="Document sections cited.")


class ExtractedFigure(BaseModel):
    """A single figure pulled from retrieved passages, for tie-out checks.

    The LLM extracts the number exactly as printed plus its stated unit -- it never does the
    raw-dollar unit conversion itself. Multiplying by the unit is deterministic Python math
    (see audit_checklist._to_raw_usd), not an LLM arithmetic step, for the same reason
    calculate_variance exists: don't trust an LLM to do arithmetic that code can do exactly.
    """

    value: float | None = Field(None, description="The figure exactly as printed in the table, e.g. 34550.")
    unit: Literal["raw", "thousands", "millions", "billions"] | None = Field(
        None, description="The unit the printed number is stated in, per the table's header."
    )
