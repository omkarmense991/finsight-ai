# src/finsight/rag/pipeline.py

import time

from src.finsight.config import (
    DEFAULT_TOP_K,
    MIN_RETRIEVAL_SCORE,
    LLM_PROVIDER,
    USE_RERANKER,
    RERANK_TOP_K,
)
from src.finsight.rag.llm_client import get_llm_client
from src.finsight.rag.prompt import FALLBACK_RESPONSE, build_user_prompt
from src.finsight.rag.reranker import CrossEncoderReranker
from src.finsight.rag.retriever import MultiQueryRetriever
from src.finsight.schemas import RetrievedChunk

FALLBACK_ANSWER = FALLBACK_RESPONSE

RETRIEVAL_STRATEGY = (
    "hybrid_faiss_bm25_rrf_blended_rerank" if USE_RERANKER else "hybrid_faiss_bm25_rrf"
)


class RAGPipeline:
    def __init__(self):
        self.retriever = MultiQueryRetriever()
        self.reranker = CrossEncoderReranker() if USE_RERANKER else None
        self.llm_client = get_llm_client()

    def _elapsed_ms(self, start_time: float) -> float:
        return round((time.perf_counter() - start_time) * 1000, 2)

    def _build_timings(
        self,
        retrieval_ms: float = 0.0,
        rerank_ms: float = 0.0,
        llm_ms: float = 0.0,
        total_ms: float = 0.0,
    ) -> dict:
        return {
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "llm_ms": llm_ms,
            "total_ms": total_ms,
        }

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
        best_retrieval_score: float | None = None,
        best_final_score: float | None = None,
        timings_ms: dict | None = None,
    ) -> dict:
        return {
            "top_k": top_k,
            "min_score": min_score,
            "best_score": best_score,
            "best_retrieval_score": best_retrieval_score,
            "best_final_score": best_final_score,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_strategy": RETRIEVAL_STRATEGY,
            "llm_provider": LLM_PROVIDER,
            "is_answer_found": is_answer_found,
            "fallback_reason": fallback_reason,
            "timings_ms": timings_ms,
        }

    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_RETRIEVAL_SCORE,
    ) -> dict:
        total_start = time.perf_counter()

        retrieval_ms = 0.0
        rerank_ms = 0.0
        llm_ms = 0.0

        candidate_top_k = RERANK_TOP_K if USE_RERANKER else top_k

        retrieval_start = time.perf_counter()

        candidate_chunks = self.retriever.retrieve(
            question=question,
            top_k=candidate_top_k,
        )

        retrieval_ms = self._elapsed_ms(retrieval_start)

        if not candidate_chunks:
            total_ms = self._elapsed_ms(total_start)

            timings_ms = self._build_timings(
                retrieval_ms=retrieval_ms,
                rerank_ms=rerank_ms,
                llm_ms=llm_ms,
                total_ms=total_ms,
            )

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
                    best_retrieval_score=None,
                    best_final_score=None,
                    timings_ms=timings_ms,
                ),
            }

        # This score is from the original hybrid retriever.
        # Use it only for the retrieval confidence threshold.
        best_retrieval_score = candidate_chunks[0].score

        if best_retrieval_score < min_score:
            total_ms = self._elapsed_ms(total_start)

            timings_ms = self._build_timings(
                retrieval_ms=retrieval_ms,
                rerank_ms=rerank_ms,
                llm_ms=llm_ms,
                total_ms=total_ms,
            )

            return {
                "question": question,
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "metadata": self._build_metadata(
                    top_k=top_k,
                    min_score=min_score,
                    best_score=best_retrieval_score,
                    retrieved_chunks=len(candidate_chunks),
                    is_answer_found=False,
                    fallback_reason="retrieval_score_below_threshold",
                    best_retrieval_score=best_retrieval_score,
                    best_final_score=None,
                    timings_ms=timings_ms,
                ),
            }

        if self.reranker is not None:
            rerank_start = time.perf_counter()

            final_chunks = self.reranker.rerank(
                question=question,
                chunks=candidate_chunks,
                top_k=top_k,
            )

            rerank_ms = self._elapsed_ms(rerank_start)
        else:
            final_chunks = candidate_chunks[:top_k]

        # This score is the final score shown in the sources array.
        # After reranking, this becomes the blended reranker score.
        best_final_score = final_chunks[0].score if final_chunks else None

        context = self._format_context(final_chunks)
        user_prompt = build_user_prompt(question=question, context=context)

        llm_start = time.perf_counter()

        answer = self.llm_client.generate_answer(user_prompt).strip()

        llm_ms = self._elapsed_ms(llm_start)

        total_ms = self._elapsed_ms(total_start)

        timings_ms = self._build_timings(
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
        )

        if FALLBACK_ANSWER.lower() in answer.lower():
            return {
                "question": question,
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "metadata": self._build_metadata(
                    top_k=top_k,
                    min_score=min_score,
                    best_score=best_final_score,
                    retrieved_chunks=len(final_chunks),
                    is_answer_found=False,
                    fallback_reason="llm_could_not_answer_from_context",
                    best_retrieval_score=best_retrieval_score,
                    best_final_score=best_final_score,
                    timings_ms=timings_ms,
                ),
            }

        return {
            "question": question,
            "answer": answer,
            "sources": self._build_sources(final_chunks),
            "metadata": self._build_metadata(
                top_k=top_k,
                min_score=min_score,
                best_score=best_final_score,
                retrieved_chunks=len(final_chunks),
                is_answer_found=True,
                best_retrieval_score=best_retrieval_score,
                best_final_score=best_final_score,
                timings_ms=timings_ms,
            ),
        }
