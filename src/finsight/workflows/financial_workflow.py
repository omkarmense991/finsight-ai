# src/finsight/workflows/financial_workflow.py

from typing import Any

from langgraph.graph import END, StateGraph

from src.finsight.workflows.state import FinancialWorkflowState
from src.finsight.workflows.tools import (
    financial_highlights_tool,
    rag_answer_tool,
)


class FinancialWorkflow:
    def __init__(
        self,
        rag_pipeline,
        financial_extractor,
    ):
        self.rag_pipeline = rag_pipeline
        self.financial_extractor = financial_extractor
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(FinancialWorkflowState)

        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("rag_answer_tool", self._run_rag_answer_tool)
        graph.add_node(
            "financial_highlights_tool",
            self._run_financial_highlights_tool,
        )

        graph.set_entry_point("classify_intent")

        graph.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "normal_qna": "rag_answer_tool",
                "financial_extraction": "financial_highlights_tool",
            },
        )

        graph.add_edge("rag_answer_tool", END)
        graph.add_edge("financial_highlights_tool", END)

        return graph.compile()

    def _classify_intent(
        self,
        state: FinancialWorkflowState,
    ) -> FinancialWorkflowState:
        question = state["question"]
        lower_question = question.lower()

        extraction_keywords = [
            "extract",
            "financial highlights",
            "structured",
            "json",
            "financial metrics",
            "key financials",
            "financial summary",
        ]

        is_financial_extraction = any(
            keyword in lower_question
            for keyword in extraction_keywords
        )

        intent = (
            "financial_extraction"
            if is_financial_extraction
            else "normal_qna"
        )

        return {
            **state,
            "intent": intent,
        }

    def _route_by_intent(
        self,
        state: FinancialWorkflowState,
    ) -> str:
        return state["intent"]

    def _run_rag_answer_tool(
        self,
        state: FinancialWorkflowState,
    ) -> FinancialWorkflowState:
        result = rag_answer_tool(
            question=state["question"],
            top_k=state.get("top_k", 5),
            rag_pipeline=self.rag_pipeline,
        )

        return {
            **state,
            "tool_called": "rag_answer_tool",
            "result": result,
        }

    def _run_financial_highlights_tool(
        self,
        state: FinancialWorkflowState,
    ) -> FinancialWorkflowState:
        result = financial_highlights_tool(
            top_k=state.get("top_k", 12),
            financial_extractor=self.financial_extractor,
        )

        return {
            **state,
            "tool_called": "financial_highlights_tool",
            "result": result,
        }

    def invoke(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        final_state = self.graph.invoke(
            {
                "question": question,
                "top_k": top_k,
            }
        )

        return {
            "question": final_state["question"],
            "intent": final_state["intent"],
            "tool_called": final_state["tool_called"],
            "result": final_state["result"],
        }