# src/finsight/embeddings/embedder.py
import numpy as np
from sentence_transformers import SentenceTransformer

from src.finsight.config import EMBEDDING_MODEL_NAME


class TextEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return np.array(embeddings, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        return np.array(embedding, dtype="float32")
