# src/finsight/rag/pipeline.py

from src.finsight.config import (
    DEFAULT_TOP_K,
    MIN_RETRIEVAL_SCORE,
    LLM_PROVIDER,
)
from src.finsight.rag.llm_client import get_llm_client
from src.finsight.rag.prompt import build_user_prompt
from src.finsight.rag.retriever import MultiQueryRetriever
from src.finsight.schemas import RetrievedChunk

FALLBACK_ANSWER = "I could not find this information in the uploaded documents."
RETRIEVAL_STRATEGY = "hybrid_faiss_bm25_rrf"


class RAGPipeline:
    def __init__(self):
        self.retriever = MultiQueryRetriever()
        self.llm_client = get_llm_client()

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        context_parts = []

        for chunk in chunks:
            context_parts.append(f"""
[Source: {chunk.document_name}, Page: {chunk.page_number}, Chunk: {chunk.chunk_id}, Score: {chunk.score:.4f}]
{chunk.text}
""".strip())

        return "\n\n".join(context_parts)

    def _build_sources(self, chunks: list[RetrievedChunk]) -> list[dict]:
        return [
            {
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
            }
            for chunk in chunks
        ]

    def _build_metadata(
        self,
        top_k: int,
        min_score: float,
        best_score: float | None,
        retrieved_chunks: int,
        is_answer_found: bool,
        fallback_reason: str | None = None,
    ) -> dict:
        return {
            "top_k": top_k,
            "min_score": min_score,
            "best_score": best_score,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_strategy": RETRIEVAL_STRATEGY,
            "llm_provider": LLM_PROVIDER,
            "is_answer_found": is_answer_found,
            "fallback_reason": fallback_reason,
        }

    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_RETRIEVAL_SCORE,
    ) -> dict:
        retrieved_chunks = self.retriever.retrieve(
            question=question,
            top_k=top_k,
        )

        if not retrieved_chunks:
            return {
                "question": question,
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "metadata": self._build_metadata(
                    top_k=top_k,
                    min_score=min_score,
                    best_score=None,
                    retrieved_chunks=0,
                    is_answer_found=False,
                    fallback_reason="no_chunks_retrieved",
                ),
            }

        best_score = retrieved_chunks[0].score

        if best_score < min_score:
            return {
                "question": question,
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "metadata": self._build_metadata(
                    top_k=top_k,
                    min_score=min_score,
                    best_score=best_score,
                    retrieved_chunks=len(retrieved_chunks),
                    is_answer_found=False,
                    fallback_reason="retrieval_score_below_threshold",
                ),
            }

        context = self._format_context(retrieved_chunks)
        user_prompt = build_user_prompt(question=question, context=context)

        answer = self.llm_client.generate_answer(user_prompt).strip()

        if FALLBACK_ANSWER.lower() in answer.lower():
            return {
                "question": question,
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "metadata": self._build_metadata(
                    top_k=top_k,
                    min_score=min_score,
                    best_score=best_score,
                    retrieved_chunks=len(retrieved_chunks),
                    is_answer_found=False,
                    fallback_reason="llm_could_not_answer_from_context",
                ),
            }

        return {
            "question": question,
            "answer": answer,
            "sources": self._build_sources(retrieved_chunks),
            "metadata": self._build_metadata(
                top_k=top_k,
                min_score=min_score,
                best_score=best_score,
                retrieved_chunks=len(retrieved_chunks),
                is_answer_found=True,
            ),
        }
