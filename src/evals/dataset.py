"""Ground-truth Q&A pairs for evaluating the audit agent against the Apple 10-K."""
from pydantic import BaseModel


class EvalCase(BaseModel):
    question: str
    expected_answer: str


EVAL_DATASET: list[EvalCase] = [
    EvalCase(
        question="What was Apple's total net sales for fiscal year 2025?",
        expected_answer="Apple's total net sales for fiscal year 2025 were $416,161 million.",
    ),
    EvalCase(
        question="How much did Apple's total net sales grow from fiscal 2024 to fiscal 2025?",
        expected_answer=(
            "Total net sales grew about 6%, from $391,035 million in fiscal 2024 to "
            "$416,161 million in fiscal 2025, an increase of $25,126 million."
        ),
    ),
    EvalCase(
        question="What was Apple's total gross margin percentage in fiscal year 2025?",
        expected_answer="Apple's total gross margin percentage in fiscal year 2025 was 46.9%.",
    ),
    EvalCase(
        question="What were Apple's iPhone net sales in fiscal year 2025?",
        expected_answer=(
            "iPhone net sales were $209,586 million in fiscal 2025, up 4% from "
            "$201,183 million in fiscal 2024."
        ),
    ),
    EvalCase(
        question="What was Apple's operating income for fiscal year 2025?",
        expected_answer="Apple's operating income for fiscal year 2025 was $133,050 million.",
    ),
    EvalCase(
        question="How much did Apple spend on research and development in fiscal 2025?",
        expected_answer=(
            "Apple spent $34,550 million on research and development in fiscal 2025, "
            "up 10% from $31,370 million in fiscal 2024."
        ),
    ),
]
