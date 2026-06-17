# src/finsight/rag/retriever.py

from collections import defaultdict

from src.finsight.config import INDEX_DIR, DEFAULT_TOP_K
from src.finsight.embeddings.embedder import TextEmbedder
from src.finsight.schemas import RetrievedChunk
from src.finsight.vector_store.faiss_store import FaissVectorStore
from src.finsight.rag.query_expander import generate_search_queries


class MultiQueryRetriever:
    def __init__(self):
        self.embedder = TextEmbedder()

        self.vector_store = FaissVectorStore(
            index_path=INDEX_DIR / "finsight.faiss",
            metadata_path=INDEX_DIR / "chunks.json",
        )

        self.vector_store.load()

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        search_queries = generate_search_queries(question)

        chunk_lookup = {}
        fusion_scores = defaultdict(float)
        max_vector_scores = defaultdict(float)
        matched_query_count = defaultdict(int)

        internal_top_k = max(top_k, 10)
        rrf_constant = 60

        for search_query in search_queries:
            query_embedding = self.embedder.embed_query(search_query)

            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=internal_top_k,
            )

            for rank, chunk in enumerate(results, start=1):
                chunk_lookup[chunk.chunk_id] = chunk

                fusion_scores[chunk.chunk_id] += 1 / (rrf_constant + rank)

                max_vector_scores[chunk.chunk_id] = max(
                    max_vector_scores[chunk.chunk_id],
                    chunk.score,
                )

                matched_query_count[chunk.chunk_id] += 1

        ranked_chunk_ids = sorted(
            fusion_scores.keys(),
            key=lambda chunk_id: (
                fusion_scores[chunk_id],
                matched_query_count[chunk_id],
                max_vector_scores[chunk_id],
            ),
            reverse=True,
        )

        final_chunks = []

        for chunk_id in ranked_chunk_ids[:top_k]:
            chunk = chunk_lookup[chunk_id]

            final_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    score=float(fusion_scores[chunk_id]),
                )
            )

        return final_chunks