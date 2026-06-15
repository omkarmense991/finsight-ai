# FinSight AI — Financial Document Intelligence Assistant

FinSight AI is a production-style AI Engineering project focused on financial document intelligence using Retrieval-Augmented Generation, LLMs, embeddings, vector search, and source-grounded question answering.

The goal is to allow users to upload financial documents such as annual reports, earnings call transcripts, investor presentations, SEBI/RBI circulars, and company filings, then ask questions grounded in the uploaded documents.

## Planned Features

- PDF document ingestion
- Text extraction
- Chunking
- Embedding generation
- Vector search using FAISS/Chroma
- Source-grounded question answering
- Citations with document and page references
- FastAPI backend
- Metadata storage
- RAG evaluation
- Prompt engineering
- Structured financial extraction
- Controlled agentic workflows
- Dockerized deployment

## Current Stage

Stage 1: Basic Document Ingestion and RAG MVP

## Tech Stack

- Python
- PyMuPDF / pdfplumber
- Sentence Transformers / OpenAI embeddings
- FAISS / Chroma
- OpenAI / Gemini / Anthropic / local LLMs
- FastAPI
- PostgreSQL
- Docker