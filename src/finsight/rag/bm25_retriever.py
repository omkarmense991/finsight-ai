# src/finsight/rag/bm25_retriever.py

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.finsight.config import INDEX_DIR
from src.finsight.schemas import RetrievedChunk


class BM25Retriever:
    def __init__(
        self,
        metadata_path: Path = INDEX_DIR / "chunks.json",
    ):
        self.metadata_path = metadata_path
        self.chunks = []
        self.bm25 = None
        self.tokenized_corpus = []

        self.load()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def load(self) -> None:
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"BM25 metadata file not found: {self.metadata_path}"
            )

        with open(self.metadata_path, "r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        texts = [chunk["text"] for chunk in self.chunks]
        self.tokenized_corpus = [self._tokenize(text) for text in texts]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self.bm25 is None:
            raise ValueError("BM25 index is not loaded.")

        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            score = float(scores[index])

            if score <= 0:
                continue

            chunk = self.chunks[index]

            results.append(
                RetrievedChunk(
                    document_name=chunk["document_name"],
                    page_number=chunk["page_number"],
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    score=score,
                )
            )

        return results
