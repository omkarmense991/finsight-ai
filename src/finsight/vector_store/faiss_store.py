# src/finsight/vector_store/faiss_store.py
import json
from pathlib import Path

import faiss
import numpy as np

from src.finsight.schemas import DocumentChunk, RetrievedChunk


class FaissVectorStore:
    def __init__(self, index_path: Path, metadata_path: Path):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.chunks: list[DocumentChunk] = []

    def build(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        self.chunks = chunks

    def save(self) -> None:
        if self.index is None:
            raise ValueError("Index has not been built.")

        faiss.write_index(self.index, str(self.index_path))

        metadata = [
            {
                "chunk_id": chunk.chunk_id,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "text": chunk.text,
            }
            for chunk in self.chunks
        ]

        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        self.index = faiss.read_index(str(self.index_path))

        with open(self.metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        self.chunks = [
            DocumentChunk(
                chunk_id=item["chunk_id"],
                document_name=item["document_name"],
                page_number=item["page_number"],
                text=item["text"],
            )
            for item in metadata
        ]

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[RetrievedChunk]:
        if self.index is None:
            raise ValueError("Index has not been loaded or built.")

        scores, indices = self.index.search(query_embedding, top_k)

        results: list[RetrievedChunk] = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            chunk = self.chunks[index]

            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    score=float(score),
                )
            )

        return results
