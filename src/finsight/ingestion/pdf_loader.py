# src/finsight/ingestion/pdf_loader.py
from pathlib import Path
import fitz

from src.finsight.schemas import PageText


def load_pdf_pages(pdf_path: str | Path) -> list[PageText]:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = fitz.open(pdf_path)
    pages: list[PageText] = []

    for page_index, page in enumerate(document):
        text = page.get_text("text", sort=True).strip()

        if not text:
            continue

        pages.append(
            PageText(
                document_name=pdf_path.name,
                page_number=page_index + 1,
                text=text,
            )
        )

    return pages
