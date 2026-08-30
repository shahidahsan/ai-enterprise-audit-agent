"""Tests for the DeepEval evaluation harness and JSON report generator."""
import json
from unittest.mock import MagicMock

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from langchain_core.documents import Document

from src.agent.schemas import AuditFinding
from src.evals.dataset import EvalCase
from src.evals.runner import aggregate_scores, build_test_case, run_evaluation, score_test_case


def _fake_metric(cls, score: float, reason: str = "ok"):
    metric = MagicMock(spec=cls)
    metric.measure.return_value = score
    metric.score = score
    metric.reason = reason
    return metric


def _fake_store(docs):
    store = MagicMock()
    store.similarity_search.return_value = docs
    return store


# --- build_test_case -----------------------------------------------------------


def test_build_test_case_packages_agent_output_and_retrieval_context(mocker):
    case = EvalCase(question="What was net sales?", expected_answer="$416,161 million.")
    finding = AuditFinding(
        summary="Net sales were $416,161 million.",
        data_points=["416,161 million"],
        compliance_status="PASSED",
        sources=["Item 7"],
    )
    mocker.patch("src.evals.runner.run_audit_query", return_value=finding)
    store = _fake_store([Document(page_content="Total net sales $416,161", metadata={"page": 26})])

    test_case = build_test_case(case, store)

    assert test_case.input == "What was net sales?"
    assert "416,161 million" in test_case.actual_output
    assert test_case.expected_output == "$416,161 million."
    assert test_case.retrieval_context == ["Total net sales $416,161"]


# --- score_test_case -------------------------------------------------------------


def test_score_test_case_returns_score_and_reason_per_metric():
    fake_test_case = MagicMock()
    metrics = [
        _fake_metric(FaithfulnessMetric, 0.9, "faithful"),
        _fake_metric(AnswerRelevancyMetric, 0.8, "relevant"),
    ]

    scores = score_test_case(fake_test_case, metrics)

    assert scores == {
        "FaithfulnessMetric": {"score": 0.9, "reason": "faithful"},
        "AnswerRelevancyMetric": {"score": 0.8, "reason": "relevant"},
    }
    for metric in metrics:
        metric.measure.assert_called_once_with(fake_test_case)


# --- aggregate_scores -------------------------------------------------------------


def test_aggregate_scores_averages_each_metric_across_queries():
    per_query = [
        {"question": "q1", "scores": {"FaithfulnessMetric": {"score": 1.0, "reason": "a"}}},
        {"question": "q2", "scores": {"FaithfulnessMetric": {"score": 0.0, "reason": "b"}}},
    ]

    assert aggregate_scores(per_query) == {"FaithfulnessMetric": 0.5}


def test_aggregate_scores_handles_empty_dataset():
    assert aggregate_scores([]) == {}


# --- run_evaluation (end to end, everything mocked) -------------------------------


def test_run_evaluation_writes_json_report(tmp_path, mocker):
    dataset = [
        EvalCase(question="What was net sales?", expected_answer="$416,161 million."),
        EvalCase(question="What was gross margin?", expected_answer="46.9%."),
    ]
    finding = AuditFinding(summary="An answer.", data_points=[], compliance_status="PASSED", sources=[])
    mocker.patch("src.evals.runner.run_audit_query", return_value=finding)
    store = _fake_store([Document(page_content="some context", metadata={})])
    metrics = [_fake_metric(FaithfulnessMetric, 0.75, "reason")]
    output_path = tmp_path / "eval_results.json"

    report = run_evaluation(dataset=dataset, store=store, metrics=metrics, output_path=output_path)

    assert output_path.exists()
    on_disk = json.loads(output_path.read_text())
    assert on_disk == report
    assert report["aggregate"] == {"FaithfulnessMetric": 0.75}
    assert len(report["per_query"]) == 2
    assert report["per_query"][0]["question"] == "What was net sales?"
