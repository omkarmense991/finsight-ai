# scripts/ingest_document.py
import sys
from pathlib import Path

from src.finsight.config import INDEX_DIR
from src.finsight.ingestion.pdf_loader import load_pdf_pages
from src.finsight.ingestion.chunker import chunk_pages
from src.finsight.embeddings.embedder import TextEmbedder
from src.finsight.vector_store.faiss_store import FaissVectorStore


def ingest_document(pdf_path: str) -> None:
    pdf_path = Path(pdf_path)

    print(f"Loading PDF: {pdf_path}")
    pages = load_pdf_pages(pdf_path)

    print(f"Extracted {len(pages)} pages with text.")

    print("Creating chunks...")
    chunks = chunk_pages(pages)

    print(f"Created {len(chunks)} chunks.")

    print("Generating embeddings...")
    embedder = TextEmbedder()
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])

    print("Building FAISS index...")
    vector_store = FaissVectorStore(
        index_path=INDEX_DIR / "finsight.faiss",
        metadata_path=INDEX_DIR / "chunks.json",
    )

    vector_store.build(chunks=chunks, embeddings=embeddings)
    vector_store.save()

    print("Ingestion complete.")
    print(f"Saved index to: {INDEX_DIR}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python -m scripts.ingest_document <pdf_path>")

    ingest_document(sys.argv[1])
