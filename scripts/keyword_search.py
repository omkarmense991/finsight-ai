# scripts/keyword_search.py

import sys
import json

from src.finsight.config import INDEX_DIR


def keyword_search(keyword: str) -> None:
    metadata_path = INDEX_DIR / "chunks.json"

    with open(metadata_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    keyword_lower = keyword.lower()

    matches = []

    for chunk in chunks:
        text_lower = chunk["text"].lower()

        if keyword_lower in text_lower:
            matches.append(chunk)

    print(f"\nFound {len(matches)} chunks containing: {keyword}\n")

    for idx, chunk in enumerate(matches[:10], start=1):
        print("=" * 100)
        print(f"Match {idx}")
        print(f"Document: {chunk['document_name']}")
        print(f"Page: {chunk['page_number']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print("-" * 100)
        print(chunk["text"][:1200])
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError('Usage: python -m scripts.keyword_search "keyword"')

    keyword = " ".join(sys.argv[1:])
    keyword_search(keyword)
