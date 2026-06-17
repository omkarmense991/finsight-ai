# src/finsight/ingestion/chunker.py
from src.finsight.schemas import PageText, DocumentChunk


def is_noisy_chunk(text: str) -> bool:
    word_count = len(text.split())
    lower_text = text.lower().strip()

    if word_count < 40:
        return True

    noisy_phrases = [
        "www.infosys.com",
        "infosys integrated annual report",
    ]

    if word_count < 80 and any(phrase in lower_text for phrase in noisy_phrases):
        return True

    return False


def split_text_into_chunks(
    text: str,
    chunk_size_words: int = 350,
    overlap_words: int = 75,
) -> list[str]:
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        start = end - overlap_words

    return chunks


def chunk_pages(
    pages: list[PageText],
    chunk_size_words: int = 350,
    overlap_words: int = 75,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []

    for page in pages:
        page_chunks = split_text_into_chunks(
            page.text,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )

        for idx, chunk_text in enumerate(page_chunks):
            if is_noisy_chunk(chunk_text):
                continue

            chunk_id = f"{page.document_name}_p{page.page_number}_c{idx + 1}"

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_name=page.document_name,
                    page_number=page.page_number,
                    text=chunk_text,
                )
            )

    return chunks
