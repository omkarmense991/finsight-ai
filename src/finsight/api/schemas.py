from typing import Any

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


class FinancialHighlightsRequest(BaseModel):
    top_k: int = Field(
        default=12,
        ge=6,
        le=18,
        description="Number of retrieved chunks to use for structured extraction.",
    )


class FinancialMetricResponse(BaseModel):
    value: str | None = None
    source_pages: list[int] = Field(default_factory=list)


class FinancialHighlightsDataResponse(BaseModel):
    consolidated_revenue_from_operations: FinancialMetricResponse
    standalone_revenue_from_operations: FinancialMetricResponse
    year_on_year_consolidated_revenue_growth: FinancialMetricResponse
    total_dividend_per_share: FinancialMetricResponse
    buyback_amount: FinancialMetricResponse
    buyback_price_per_share: FinancialMetricResponse


class FinancialHighlightsMetadataResponse(BaseModel):
    extraction_mode: str
    retrieval_strategy: str
    llm_provider: str
    is_answer_found: bool
    fallback_reason: str | None = None
    sources_used: int
    extracted_field_count: int = 0
    total_field_count: int = 0
    extracted_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    timings_ms: TimingResponse | None = None


class FinancialHighlightsResponse(BaseModel):
    document_name: str | None = None
    fiscal_year: str | None = None
    currency: str | None = None
    financial_highlights: FinancialHighlightsDataResponse
    sources: list[SourceResponse]
    metadata: FinancialHighlightsMetadataResponse


class WorkflowAskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="User question to route through the LangGraph workflow.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=18,
        description="Number of chunks to use for the selected workflow tool.",
    )


class WorkflowAskResponse(BaseModel):
    question: str
    intent: str
    tool_called: str
    result: dict[str, Any]
