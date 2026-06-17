# src/finsight/schemas.py

from dataclasses import dataclass


@dataclass
class PageText:
    document_name: str
    page_number: int
    text: str


@dataclass
class DocumentChunk:
    chunk_id: str
    document_name: str
    page_number: int
    text: str


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_name: str
    page_number: int
    text: str
    score: float
