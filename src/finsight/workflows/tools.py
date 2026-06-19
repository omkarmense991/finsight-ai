# src/finsight/workflows/tools.py

from typing import Any


def rag_answer_tool(
    question: str,
    top_k: int,
    rag_pipeline,
) -> dict[str, Any]:
    return rag_pipeline.ask(
        question=question,
        top_k=top_k,
    )


def financial_highlights_tool(
    top_k: int,
    financial_extractor,
) -> dict[str, Any]:
    extraction_top_k = max(top_k, 12)

    return financial_extractor.extract_financial_highlights(
        top_k=extraction_top_k,
    )
