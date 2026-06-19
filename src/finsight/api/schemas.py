from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="User question to ask against the uploaded financial documents.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of retrieved chunks to use as context.",
    )


class SourceResponse(BaseModel):
    document_name: str
    page_number: int
    chunk_id: str
    score: float


class TimingResponse(BaseModel):
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0


class MetadataResponse(BaseModel):
    top_k: int
    min_score: float
    best_score: float | None
    best_retrieval_score: float | None = None
    best_final_score: float | None = None
    retrieved_chunks: int
    retrieval_strategy: str
    llm_provider: str
    is_answer_found: bool
    fallback_reason: str | None = None
    timings_ms: TimingResponse | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]
    metadata: MetadataResponse


class DocumentUploadResponse(BaseModel):
    message: str
    document_name: str
    pages_extracted: int
    chunks_created: int
    index_path: str
    metadata_path: str