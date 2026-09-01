"""DeepEval evaluation harness: scores the agent against the ground-truth dataset."""
import json
from pathlib import Path

from deepeval.metrics import AnswerRelevancyMetric, ContextualRecallMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from src.agent.react_agent import run_audit_query
from src.evals.dataset import EVAL_DATASET, EvalCase
from src.indexer.vector_store import AuditVectorStore

JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_OUTPUT_PATH = Path("eval_results.json")


def build_metrics() -> list:
    """Instantiate the PRD-required metrics, judged by a cheap model to control cost."""
    return [
        FaithfulnessMetric(model=JUDGE_MODEL),
        AnswerRelevancyMetric(model=JUDGE_MODEL),
        ContextualRecallMetric(model=JUDGE_MODEL),
    ]


def build_test_case(case: EvalCase, store: AuditVectorStore) -> LLMTestCase:
    """Run the agent and retriever for one eval case and package a DeepEval test case."""
    retrieved_docs = store.similarity_search(case.question)
    finding, _steps = run_audit_query(case.question, store=store)
    actual_output = " ".join([finding.summary, *finding.data_points])

    return LLMTestCase(
        input=case.question,
        actual_output=actual_output,
        expected_output=case.expected_answer,
        retrieval_context=[doc.page_content for doc in retrieved_docs],
    )


def score_test_case(test_case: LLMTestCase, metrics: list) -> dict:
    """Measure one test case against all metrics; return per-metric score and reason."""
    scores = {}
    for metric in metrics:
        metric.measure(test_case)
        scores[metric.__class__.__name__] = {"score": metric.score, "reason": metric.reason}
    return scores


def aggregate_scores(per_query: list[dict]) -> dict:
    """Average each metric's score across all queries."""
    if not per_query:
        return {}
    metric_names = per_query[0]["scores"].keys()
    return {
        name: sum(q["scores"][name]["score"] for q in per_query) / len(per_query) for name in metric_names
    }


def run_evaluation(
    dataset: list[EvalCase] | None = None,
    store: AuditVectorStore | None = None,
    metrics: list | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict:
    """Run the full eval suite and write an eval_results.json report."""
    dataset = EVAL_DATASET if dataset is None else dataset
    store = store or AuditVectorStore()
    metrics = build_metrics() if metrics is None else metrics

    per_query = []
    for case in dataset:
        test_case = build_test_case(case, store)
        per_query.append({"question": case.question, "scores": score_test_case(test_case, metrics)})

    report = {"aggregate": aggregate_scores(per_query), "per_query": per_query}
    output_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_evaluation()
