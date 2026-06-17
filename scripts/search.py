# scripts/search.py
import sys

from src.finsight.rag.query_expander import generate_search_queries
from src.finsight.rag.retriever import MultiQueryRetriever


def search_question(question: str) -> None:
    print("\nOriginal question:")
    print(question)

    print("\nSearch queries:")
    for idx, search_query in enumerate(generate_search_queries(question), start=1):
        print(f"{idx}. {search_query}")

    retriever = MultiQueryRetriever()
    results = retriever.retrieve(question)

    print("\nRetrieved Chunks:\n")

    for idx, chunk in enumerate(results, start=1):
        print("=" * 100)
        print(f"Result {idx}")
        print(f"Document: {chunk.document_name}")
        print(f"Page: {chunk.page_number}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Fusion Score: {chunk.score:.4f}")
        print("-" * 100)
        print(chunk.text[:1200])
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError('Usage: python -m scripts.search "your question"')

    question = " ".join(sys.argv[1:])
    search_question(question)