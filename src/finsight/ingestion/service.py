# src/finsight/ingestion/service.py

from pathlib import Path

from src.finsight.config import INDEX_DIR
from src.finsight.embeddings.embedder import TextEmbedder
from src.finsight.ingestion.chunker import chunk_pages
from src.finsight.ingestion.pdf_loader import load_pdf_pages
from src.finsight.vector_store.faiss_store import FaissVectorStore


def ingest_pdf_document(pdf_path: Path) -> dict:
    pages = load_pdf_pages(pdf_path)

    if not pages:
        raise ValueError("No text pages were extracted from the PDF.")

    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError("No chunks were created from the PDF.")

    embedder = TextEmbedder()

    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_texts(texts)

    vector_store = FaissVectorStore(
        index_path=INDEX_DIR / "finsight.faiss",
        metadata_path=INDEX_DIR / "chunks.json",
    )

    vector_store.build(
        chunks=chunks,
        embeddings=embeddings,
    )

    vector_store.save()

    return {
        "document_name": pdf_path.name,
        "pages_extracted": len(pages),
        "chunks_created": len(chunks),
        "index_path": str(INDEX_DIR / "finsight.faiss"),
        "metadata_path": str(INDEX_DIR / "chunks.json"),
    }