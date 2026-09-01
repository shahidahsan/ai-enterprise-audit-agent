"""ReAct agent loop: wires the configured LLM to the audit tools via native tool calling."""
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.schemas import AuditFinding
from src.config import Settings, get_settings
from src.indexer.vector_store import AuditVectorStore
from src.observability.tracing import configure_langsmith
from src.tools.calculator_tool import calculate_variance
from src.tools.compliance_tool import compliance_flag_checker
from src.tools.search_tool import document_search

MAX_ITERATIONS = 6


class ToolStep(TypedDict):
    tool: str
    tool_input: dict
    output: str


def build_llm(settings: Settings | None = None) -> BaseChatModel:
    """Instantiate the configured chat model (OpenAI or Anthropic, pluggable via Settings)."""
    settings = settings or get_settings()
    if settings.llm_provider == "anthropic":
        return ChatAnthropic(model=settings.agent_model, api_key=settings.anthropic_api_key)
    return ChatOpenAI(model=settings.agent_model, api_key=settings.openai_api_key)


def build_tools(store: AuditVectorStore | None = None) -> list[StructuredTool]:
    """Wrap the deterministic tool functions as LangChain tools for the LLM to call."""

    def _search(query: str) -> str:
        return document_search(query, store=store)

    return [
        StructuredTool.from_function(
            func=_search,
            name="document_search",
            description="Search the indexed SEC filing for relevant passages. Args: query (str).",
        ),
        StructuredTool.from_function(
            func=calculate_variance,
            name="calculate_variance",
            description=(
                "Compute absolute change, percentage variance, and trend between two numeric "
                "values. Args: current_val (float), prior_val (float)."
            ),
        ),
        StructuredTool.from_function(
            func=compliance_flag_checker,
            name="compliance_flag_checker",
            description=(
                "Check a clause of extracted text against a predefined compliance disclosure rule. "
                "Args: clause_text (str), rule_type (str: one of revenue_recognition, risk_factor, "
                "related_party_transaction, material_weakness)."
            ),
        ),
    ]


def _run_tool_loop(
    llm_with_tools: BaseChatModel, tools_by_name: dict, messages: list[BaseMessage]
) -> tuple[str, list[ToolStep]]:
    """Repeatedly call the LLM, dispatching any requested tool calls, until it gives a final answer."""
    steps: list[ToolStep] = []
    for _ in range(MAX_ITERATIONS):
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)
        if not ai_message.tool_calls:
            return ai_message.content, steps
        for call in ai_message.tool_calls:
            tool = tools_by_name[call["name"]]
            result = tool.invoke(call["args"])
            steps.append({"tool": call["name"], "tool_input": call["args"], "output": str(result)})
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    raise RuntimeError("Agent exceeded max tool-calling iterations without a final answer")


def run_audit_query(
    query: str,
    llm: BaseChatModel | None = None,
    store: AuditVectorStore | None = None,
) -> tuple[AuditFinding, list[ToolStep]]:
    """Run the ReAct tool loop for a query, then enforce the AuditFinding output schema.

    Returns the structured finding alongside the tool-call trace (for UI display).
    """
    settings = get_settings()
    configure_langsmith(settings)
    llm = llm or build_llm(settings)

    tools = build_tools(store=store)
    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]
    final_text, steps = _run_tool_loop(llm_with_tools, tools_by_name, messages)

    structurer = llm.with_structured_output(AuditFinding)
    finding = structurer.invoke(f"Format the following audit finding into the required structure:\n\n{final_text}")
    return finding, steps
