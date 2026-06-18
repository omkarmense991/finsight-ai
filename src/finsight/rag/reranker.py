# src/finsight/rag/reranker.py

from sentence_transformers import CrossEncoder

from src.finsight.config import (
    RERANKER_HYBRID_WEIGHT,
    RERANKER_MODEL_NAME,
    RERANKER_MODEL_WEIGHT,
)
from src.finsight.schemas import RetrievedChunk


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        hybrid_weight: float = RERANKER_HYBRID_WEIGHT,
        reranker_weight: float = RERANKER_MODEL_WEIGHT,
    ):
        self.model = CrossEncoder(model_name)
        self.hybrid_weight = hybrid_weight
        self.reranker_weight = reranker_weight

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0 for _ in scores]

        return [(score - min_score) / (max_score - min_score) for score in scores]

    def rerank(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(question, chunk.text) for chunk in chunks]

        reranker_scores = self.model.predict(pairs)

        hybrid_scores = [float(chunk.score) for chunk in chunks]
        reranker_scores = [float(score) for score in reranker_scores]

        normalized_hybrid_scores = self._normalize_scores(hybrid_scores)
        normalized_reranker_scores = self._normalize_scores(reranker_scores)

        scored_chunks = []

        for chunk, hybrid_score, reranker_score in zip(
            chunks,
            normalized_hybrid_scores,
            normalized_reranker_scores,
        ):
            final_score = (
                self.hybrid_weight * hybrid_score
                + self.reranker_weight * reranker_score
            )

            reranked_chunk = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                text=chunk.text,
                score=float(final_score),
            )

            scored_chunks.append((reranked_chunk, final_score))

        scored_chunks.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        reranked_chunks = [chunk for chunk, _ in scored_chunks]

        return reranked_chunks[:top_k]
