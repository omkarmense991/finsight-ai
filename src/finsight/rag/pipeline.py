# src/finsight/rag/pipeline.py
from src.finsight.config import DEFAULT_TOP_K, MIN_RETRIEVAL_SCORE
from src.finsight.rag.llm_client import get_llm_client
from src.finsight.rag.prompt import build_user_prompt
from src.finsight.rag.retriever import MultiQueryRetriever
from src.finsight.schemas import RetrievedChunk


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
""")

        return "\n\n".join(context_parts)

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
                "answer": "I could not find this information in the uploaded documents.",
                "sources": [],
            }

        best_score = retrieved_chunks[0].score

        if best_score < min_score:
            return {
                "answer": "I could not find this information in the uploaded documents.",
                "sources": [
                    {
                        "document_name": chunk.document_name,
                        "page_number": chunk.page_number,
                        "chunk_id": chunk.chunk_id,
                        "score": chunk.score,
                    }
                    for chunk in retrieved_chunks
                ],
            }

        context = self._format_context(retrieved_chunks)
        user_prompt = build_user_prompt(question=question, context=context)

        answer = self.llm_client.generate_answer(user_prompt)

        sources = [
            {
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
            }
            for chunk in retrieved_chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
        }
