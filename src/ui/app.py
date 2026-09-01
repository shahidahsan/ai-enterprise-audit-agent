"""Two-column Streamlit dashboard: interactive audit chat + evaluation scorecard."""
import json
import time

import streamlit as st

from src.agent.react_agent import build_llm, run_audit_query
from src.agent.schemas import AuditFinding
from src.config import get_settings
from src.evals.runner import DEFAULT_OUTPUT_PATH
from src.indexer.vector_store import AuditVectorStore

st.set_page_config(page_title="Compliance & Audit Agent", layout="wide")

LANGSMITH_URL = "https://smith.langchain.com/"


@st.cache_resource
def get_store() -> AuditVectorStore:
    return AuditVectorStore()


@st.cache_resource
def get_llm():
    return build_llm()


def render_status_badge(status: str) -> None:
    if status == "PASSED":
        st.success(f"Compliance status: {status}")
    elif status == "FLAGGED":
        st.error(f"Compliance status: {status}")
    else:
        st.warning(f"Compliance status: {status}")


def render_finding(finding: AuditFinding, steps: list[dict], latency: float) -> None:
    st.markdown(f"**Summary:** {finding.summary}")
    render_status_badge(finding.compliance_status)

    if finding.data_points:
        st.markdown("**Data points:**\n" + "\n".join(f"- {point}" for point in finding.data_points))
    if finding.sources:
        st.markdown("**Sources:** " + ", ".join(finding.sources))

    st.caption(f"Answered in {latency:.2f}s using {len(steps)} tool call(s)")
    for i, step in enumerate(steps, start=1):
        with st.expander(f"\U0001f527 Step {i}: {step['tool']}"):
            st.json(step["tool_input"])
            st.text(step["output"])


def render_chat_panel() -> None:
    st.header("Interactive Audit")
    st.session_state.setdefault("history", [])

    query = st.chat_input("Ask a question about the filing...")
    if query:
        with st.spinner("Running audit agent..."):
            start = time.perf_counter()
            finding, steps = run_audit_query(query, llm=get_llm(), store=get_store())
            latency = time.perf_counter() - start
        st.session_state.history.append({"query": query, "finding": finding, "steps": steps, "latency": latency})

    for turn in reversed(st.session_state.history):
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            render_finding(turn["finding"], turn["steps"], turn["latency"])


def render_eval_panel() -> None:
    st.header("Evaluation & Reliability")
    settings = get_settings()

    if DEFAULT_OUTPUT_PATH.exists():
        report = json.loads(DEFAULT_OUTPUT_PATH.read_text())
        aggregate = report.get("aggregate", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Faithfulness", f"{aggregate.get('FaithfulnessMetric', 0) * 100:.1f}%")
        col2.metric("Answer Relevancy", f"{aggregate.get('AnswerRelevancyMetric', 0) * 100:.1f}%")
        col3.metric("Context Recall", f"{aggregate.get('ContextualRecallMetric', 0) * 100:.1f}%")
        st.caption(f"From {len(report.get('per_query', []))} eval queries. Run `uv run python -m src.evals.runner` to refresh.")
    else:
        st.info("No eval_results.json found yet. Run `uv run python -m src.evals.runner` to generate one.")

    if st.session_state.get("history"):
        st.metric("Last query latency", f"{st.session_state.history[-1]['latency']:.2f}s")

    st.divider()
    st.markdown(f"**LangSmith project:** `{settings.langchain_project}`")
    st.link_button("Open LangSmith", LANGSMITH_URL)


def main() -> None:
    st.title("Enterprise Compliance & Audit Agent")
    left, right = st.columns([2, 1])
    with left:
        render_chat_panel()
    with right:
        render_eval_panel()


main()
