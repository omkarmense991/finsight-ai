# scripts/ask.py
import sys
import json

from src.finsight.rag.pipeline import RAGPipeline


def ask_question(question: str) -> None:
    pipeline = RAGPipeline()
    result = pipeline.ask(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    print(json.dumps(result["sources"], indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError('Usage: python -m scripts.ask "your question"')

    question = " ".join(sys.argv[1:])
    ask_question(question)
