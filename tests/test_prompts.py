"""Tests for the system prompt builder, which injects which filing is currently loaded."""
from src.agent.prompts import SYSTEM_PROMPT, build_system_prompt


def test_build_system_prompt_defaults_to_base_prompt_when_no_document_given():
    assert build_system_prompt() == SYSTEM_PROMPT


def test_build_system_prompt_names_the_company_and_fiscal_year():
    prompt = build_system_prompt(company_name="NVIDIA CORPORATION", fiscal_year=2025)

    assert "NVIDIA CORPORATION" in prompt
    assert "2025" in prompt
    assert SYSTEM_PROMPT in prompt


def test_build_system_prompt_handles_company_name_without_fiscal_year():
    prompt = build_system_prompt(company_name="Apple Inc.")

    assert "Apple Inc." in prompt
