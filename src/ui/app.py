"""Two-column Streamlit dashboard: interactive audit chat + evaluation scorecard."""
import json
import tempfile
import time
from pathlib import Path

import streamlit as st

from src.agent.audit_checklist import (
    DISCLOSURE_CHECKS,
    FINANCIAL_CHECKS,
    run_disclosure_check,
    run_tie_out_check,
)
from src.agent.react_agent import build_llm, run_audit_query
from src.agent.schemas import AuditFinding
from src.config import get_settings
from src.evals.runner import DEFAULT_OUTPUT_PATH
from src.indexer.document_indexer import index_filing
from src.indexer.document_registry import DocumentRecord, DocumentRegistry
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


def render_chat_panel(document: DocumentRecord | None) -> None:
    st.session_state.setdefault("history", [])

    query = st.chat_input("Ask a question about the filing...")
    if query:
        scope = (
            {
                "document_id": document.document_id,
                "cik": document.cik,
                "company_name": document.company_name,
                "fiscal_year": document.fiscal_year,
            }
            if document
            else {}
        )
        with st.spinner("Running audit agent..."):
            start = time.perf_counter()
            finding, steps = run_audit_query(query, llm=get_llm(), store=get_store(), **scope)
            latency = time.perf_counter() - start
        st.session_state.history.append({"query": query, "finding": finding, "steps": steps, "latency": latency})

    for turn in reversed(st.session_state.history):
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            render_finding(turn["finding"], turn["steps"], turn["latency"])


def _summarize_check(result: dict) -> str:
    detail = result.get("detail")
    if isinstance(detail, dict) and "xbrl_value" in detail:
        extracted = result.get("extracted_value")
        return f"Extracted ${extracted:,.0f} vs SEC XBRL ${detail['xbrl_value']:,.0f} ({detail['difference_pct']}% diff)"
    if isinstance(detail, dict) and "elements" in detail:
        matched = sum(el["matched"] for el in detail["elements"])
        return f"{matched}/{len(detail['elements'])} required disclosure elements found"
    return str(detail)


def _run_checklist_with_progress(document: DocumentRecord) -> list[dict]:
    llm, store = get_llm(), get_store()
    checks = [("tie_out", c) for c in FINANCIAL_CHECKS] + [("disclosure", c) for c in DISCLOSURE_CHECKS]
    results = []
    with st.status(f"Running {len(checks)} audit checks...", expanded=True) as status:
        for i, (kind, check) in enumerate(checks, start=1):
            status.update(label=f"Check {i}/{len(checks)}: {check['label']}...")
            if kind == "tie_out":
                result = run_tie_out_check(
                    check,
                    llm=llm,
                    store=store,
                    fiscal_year=document.fiscal_year,
                    document_id=document.document_id,
                    cik=document.cik,
                )
            else:
                result = run_disclosure_check(check, store=store, document_id=document.document_id)
            results.append(result)
            st.write(f"**{check['label']}** -> {result['status']}")
        status.update(label="Full audit complete", state="complete")
    return results


def render_full_audit_panel(document: DocumentRecord | None) -> None:
    st.subheader("Run Full Audit")
    st.caption(
        "Runs a fixed checklist: tie-out of key figures against live SEC XBRL data, plus disclosure "
        "checks. Takes ~20-30s (several live LLM + SEC EDGAR calls, run one at a time)."
    )
    if document is None:
        st.info("Select or upload a filing in the sidebar first.")
        return
    st.session_state.setdefault("audit_running", False)

    run_clicked = st.button(
        "Running full audit..." if st.session_state.audit_running else "Run Full Audit",
        type="primary",
        disabled=st.session_state.audit_running,
    )
    if run_clicked and not st.session_state.audit_running:
        st.session_state.audit_running = True
        st.rerun()

    if st.session_state.audit_running:
        st.session_state.audit_results = _run_checklist_with_progress(document)
        st.session_state.audit_running = False
        st.rerun()

    results = st.session_state.get("audit_results")
    if not results:
        st.info("Click 'Run Full Audit' to check the filing against tie-out and disclosure rules.")
        return

    rows = [
        {
            "Check": r["label"],
            "Status": r["status"],
            "Citation": r.get("citation") or "-",
            "Detail": _summarize_check(r),
        }
        for r in results
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


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


def _index_uploaded_file(uploaded_file) -> None:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)
    with st.sidebar.status("Indexing filing...", expanded=True):
        record = index_filing(tmp_path, llm=get_llm(), store=get_store())
    st.sidebar.success(f"Indexed {record.company_name} FY{record.fiscal_year}")
    st.session_state.selected_document_id = record.document_id


def render_sidebar() -> DocumentRecord | None:
    st.sidebar.header("Filing")
    documents = DocumentRegistry().list_documents()

    selected = None
    if documents:
        options = {f"{d.company_name} — FY{d.fiscal_year}": d for d in documents}
        labels = list(options.keys())
        default_id = st.session_state.get("selected_document_id")
        default_label = next((label for label, d in options.items() if d.document_id == default_id), labels[0])
        chosen_label = st.sidebar.selectbox("Select an indexed filing", labels, index=labels.index(default_label))
        selected = options[chosen_label]
        st.session_state.selected_document_id = selected.document_id
    else:
        st.sidebar.info("No filings indexed yet -- upload one below.")

    st.sidebar.divider()
    st.sidebar.subheader("Upload a new 10-K")
    uploaded_file = st.sidebar.file_uploader("PDF file", type="pdf")
    if uploaded_file and st.sidebar.button("Index this filing"):
        _index_uploaded_file(uploaded_file)
        st.rerun()

    return selected


def main() -> None:
    st.title("Enterprise Compliance & Audit Agent")
    selected_document = render_sidebar()
    left, right = st.columns([2, 1])
    with left:
        chat_tab, audit_tab = st.tabs(["Interactive Audit", "Run Full Audit"])
        with chat_tab:
            render_chat_panel(selected_document)
        with audit_tab:
            render_full_audit_panel(selected_document)
    with right:
        render_eval_panel()


main()
