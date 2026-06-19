# src/finsight/api/main.py

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.finsight.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentUploadResponse,
    FinancialHighlightsRequest,
    FinancialHighlightsResponse,
    WorkflowAskRequest,
    WorkflowAskResponse,
)

from src.finsight.config import RAW_DATA_DIR
from src.finsight.ingestion.service import ingest_pdf_document
from src.finsight.rag.pipeline import RAGPipeline
from src.finsight.rag.extraction import FinancialHighlightsExtractor
from src.finsight.workflows.financial_workflow import FinancialWorkflow

app = FastAPI(
    title="FinSight AI API",
    description="Financial Document Intelligence Assistant using RAG.",
    version="0.1.0",
)


app.state.rag_pipeline = RAGPipeline()
app.state.financial_extractor = FinancialHighlightsExtractor(app.state.rag_pipeline)
app.state.financial_workflow = FinancialWorkflow(
    rag_pipeline=app.state.rag_pipeline,
    financial_extractor=app.state.financial_extractor,
)


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "service": "FinSight AI API",
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> dict:
    try:
        result = app.state.rag_pipeline.ask(
            question=request.question,
            top_k=request.top_k,
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


@app.post(
    "/extract/financial-highlights",
    response_model=FinancialHighlightsResponse,
)
def extract_financial_highlights(
    request: FinancialHighlightsRequest,
) -> dict:
    try:
        result = app.state.financial_extractor.extract_financial_highlights(
            top_k=request.top_k,
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


@app.post("/documents/upload", response_model=DocumentUploadResponse)
def upload_document(file: UploadFile = File(...)) -> dict:
    try:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file must have a filename.",
            )

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported.",
            )

        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        safe_filename = Path(file.filename).name
        saved_path = RAW_DATA_DIR / safe_filename

        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ingestion_result = ingest_pdf_document(saved_path)

        app.state.rag_pipeline = RAGPipeline()
        app.state.financial_extractor = FinancialHighlightsExtractor(
            app.state.rag_pipeline
        )
        app.state.financial_workflow = FinancialWorkflow(
            rag_pipeline=app.state.rag_pipeline,
            financial_extractor=app.state.financial_extractor,
        )

        return {
            "message": "Document uploaded and indexed successfully.",
            **ingestion_result,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    finally:
        file.file.close()


@app.post(
    "/workflow/ask",
    response_model=WorkflowAskResponse,
)
def workflow_ask(request: WorkflowAskRequest) -> dict:
    try:
        result = app.state.financial_workflow.invoke(
            question=request.question,
            top_k=request.top_k,
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
