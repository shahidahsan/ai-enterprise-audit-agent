"""Tests for the ReAct agent core: schemas, tool wiring, and the tool-calling loop."""
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from src.agent.react_agent import _run_tool_loop, build_llm, build_tools, run_audit_query
from src.agent.schemas import AuditFinding

# --- AuditFinding schema ------------------------------------------------------


def test_audit_finding_accepts_each_valid_compliance_status():
    for status in ("PASSED", "FLAGGED", "NEEDS_REVIEW"):
        finding = AuditFinding(summary="ok", compliance_status=status)
        assert finding.compliance_status == status


def test_audit_finding_rejects_invalid_compliance_status():
    with pytest.raises(ValidationError):
        AuditFinding(summary="ok", compliance_status="MAYBE")


def test_audit_finding_defaults_lists_to_empty():
    finding = AuditFinding(summary="ok", compliance_status="PASSED")

    assert finding.data_points == []
    assert finding.sources == []


# --- build_llm provider selection ---------------------------------------------


def test_build_llm_selects_chat_openai_for_openai_provider(mocker, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("AGENT_MODEL", "gpt-4o-mini")
    mock_chat_openai = mocker.patch("src.agent.react_agent.ChatOpenAI")

    build_llm()

    mock_chat_openai.assert_called_once()
    assert mock_chat_openai.call_args.kwargs["model"] == "gpt-4o-mini"


def test_build_llm_selects_chat_anthropic_for_anthropic_provider(mocker, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("AGENT_MODEL", "claude-sonnet-5")
    mock_chat_anthropic = mocker.patch("src.agent.react_agent.ChatAnthropic")

    build_llm()

    mock_chat_anthropic.assert_called_once()
    assert mock_chat_anthropic.call_args.kwargs["model"] == "claude-sonnet-5"


# --- build_tools ---------------------------------------------------------------


def test_build_tools_returns_the_four_named_tools():
    tools = build_tools()

    names = {tool.name for tool in tools}
    assert names == {"document_search", "calculate_variance", "compliance_flag_checker", "verify_against_xbrl"}


def test_build_tools_wires_injected_store_into_document_search():
    fake_store = MagicMock()
    fake_store.similarity_search.return_value = []

    tools = build_tools(store=fake_store)
    search_tool = next(t for t in tools if t.name == "document_search")
    search_tool.invoke({"query": "revenue"})

    fake_store.similarity_search.assert_called_once_with("revenue")


# --- _run_tool_loop -------------------------------------------------------------


def test_run_tool_loop_invokes_correct_tool_and_returns_final_text():
    tool_call_message = AIMessage(
        content="",
        tool_calls=[{"name": "calculate_variance", "args": {"current_val": 110, "prior_val": 100}, "id": "call_1"}],
    )
    final_message = AIMessage(content="Net sales increased 10%.", tool_calls=[])
    llm_with_tools = MagicMock()
    llm_with_tools.invoke.side_effect = [tool_call_message, final_message]

    fake_tool = MagicMock()
    fake_tool.invoke.return_value = {"absolute_change": 10, "percentage_variance": 10.0, "trend": "increase"}
    tools_by_name = {"calculate_variance": fake_tool}

    result, steps = _run_tool_loop(llm_with_tools, tools_by_name, messages=[])

    fake_tool.invoke.assert_called_once_with({"current_val": 110, "prior_val": 100})
    assert result == "Net sales increased 10%."
    assert steps == [
        {
            "tool": "calculate_variance",
            "tool_input": {"current_val": 110, "prior_val": 100},
            "output": str({"absolute_change": 10, "percentage_variance": 10.0, "trend": "increase"}),
        }
    ]


def test_run_tool_loop_raises_after_exceeding_max_iterations():
    looping_message = AIMessage(
        content="",
        tool_calls=[{"name": "calculate_variance", "args": {"current_val": 1, "prior_val": 1}, "id": "call_x"}],
    )
    llm_with_tools = MagicMock()
    llm_with_tools.invoke.side_effect = [looping_message] * 20
    fake_tool = MagicMock()
    fake_tool.invoke.return_value = {}

    with pytest.raises(RuntimeError):
        _run_tool_loop(llm_with_tools, {"calculate_variance": fake_tool}, messages=[])


# --- run_audit_query (full loop + structured output guardrail) -----------------


def test_run_audit_query_returns_structured_output(make_mock_llm):
    tool_call_message = AIMessage(
        content="",
        tool_calls=[{"name": "calculate_variance", "args": {"current_val": 416161, "prior_val": 391035}, "id": "c1"}],
    )
    final_message = AIMessage(content="Net sales grew 6.43%.", tool_calls=[])
    expected = AuditFinding(
        summary="Net sales grew 6.43% year over year.",
        data_points=["Net sales: $416,161M", "Prior year: $391,035M"],
        compliance_status="PASSED",
        sources=["Item 7"],
    )
    mock_llm = make_mock_llm(tool_call_responses=[tool_call_message, final_message], structured_result=expected)

    result, steps = run_audit_query("How did net sales change?", llm=mock_llm)

    assert result == expected
    assert steps[0]["tool"] == "calculate_variance"
    mock_llm.with_structured_output.assert_called_once_with(AuditFinding)
