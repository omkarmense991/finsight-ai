# src/finsight/workflows/state.py

from typing import Any, Literal, TypedDict


WorkflowIntent = Literal[
    "normal_qna",
    "financial_extraction",
]


class FinancialWorkflowState(TypedDict, total=False):
    question: str
    top_k: int
    intent: WorkflowIntent
    tool_called: str
    result: dict[str, Any]