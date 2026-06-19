# src/finsight/rag/extraction.py

import json
import re
import time
from typing import Any

from src.finsight.config import LLM_PROVIDER, RERANK_TOP_K
from src.finsight.schemas import RetrievedChunk

EXTRACTION_RETRIEVAL_STRATEGY = "coverage_first_hybrid_faiss_bm25_rrf"

EXTRACTION_SYSTEM_PROMPT = """
You are FinSight AI's structured financial extraction engine.

Your job is to extract financial highlights from retrieved annual-report context.

Rules:
1. Use only the provided context.
2. Return only valid JSON.
3. Do not use markdown.
4. Do not wrap the JSON in ```json fences.
5. Do not add explanations before or after the JSON.
6. If a value is not clearly present in the context, use null.
7. For every extracted value, include source_pages as a list of page numbers.
8. Use exact numbers, currency symbols, units, and fiscal years from the context.
9. Do not guess or infer unsupported values.
10. If the document text uses ` before Indian currency amounts, render it as ₹.
""".strip()


EXTRACTION_QUESTION = """
Extract the following financial highlights from the annual report:

1. consolidated revenue from operations
2. standalone revenue from operations
3. year-on-year consolidated revenue growth
4. total dividend per share
5. buyback amount
6. buyback price per share

Return the answer as structured JSON only.
""".strip()


FINANCIAL_HIGHLIGHT_KEYS = [
    "consolidated_revenue_from_operations",
    "standalone_revenue_from_operations",
    "year_on_year_consolidated_revenue_growth",
    "total_dividend_per_share",
    "buyback_amount",
    "buyback_price_per_share",
]


FINANCIAL_HIGHLIGHT_QUERIES = {
    "consolidated_revenue_from_operations": [
        "consolidated revenue from operations fiscal 2026",
        "consolidated revenue from operations year ended March 31 2026",
        "revenue from operations consolidated fiscal 2026",
    ],
    "standalone_revenue_from_operations": [
        "standalone revenue from operations fiscal 2026",
        "standalone revenue from operations year ended March 31 2026",
        "revenue from operations standalone fiscal 2026",
    ],
    "year_on_year_consolidated_revenue_growth": [
        "year on year consolidated revenue growth fiscal 2026",
        "revenue growth fiscal 2026 compared with fiscal 2025",
        "consolidated revenue growth percentage fiscal 2026",
    ],
    "total_dividend_per_share": [
        "total dividend per share fiscal 2026",
        "interim dividend final dividend total dividend per share",
        "dividend fiscal 2026 48 per share",
    ],
    "buyback_amount": [
        "buyback amount fiscal 2026",
        "share buyback amount fiscal 2026",
        "maximum buyback amount rupees crore",
    ],
    "buyback_price_per_share": [
        "buyback price per share fiscal 2026",
        "maximum buyback price per equity share",
        "buyback price rupees per share",
    ],
}


class FinancialHighlightsExtractor:
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline

    def _normalize_metric_value(self, value: Any) -> str | None:
        if value is None:
            return None

        value_text = str(value).strip()

        if not value_text:
            return None

        value_text = value_text.replace("`", "₹")

        return value_text

    def _build_extraction_quality(self, extraction: dict) -> dict:
        highlights = extraction.get("financial_highlights", {})

        extracted_fields = []
        missing_fields = []

        for key in FINANCIAL_HIGHLIGHT_KEYS:
            metric = highlights.get(key, {})

            if metric.get("value") is not None:
                extracted_fields.append(key)
            else:
                missing_fields.append(key)

        return {
            "extracted_field_count": len(extracted_fields),
            "total_field_count": len(FINANCIAL_HIGHLIGHT_KEYS),
            "extracted_fields": extracted_fields,
            "missing_fields": missing_fields,
        }

    def _elapsed_ms(self, start_time: float) -> float:
        return round((time.perf_counter() - start_time) * 1000, 2)

    def _build_timings(
        self,
        retrieval_ms: float = 0.0,
        rerank_ms: float = 0.0,
        llm_ms: float = 0.0,
        total_ms: float = 0.0,
    ) -> dict:
        return {
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "llm_ms": llm_ms,
            "total_ms": total_ms,
        }

    def _build_sources(self, chunks: list[RetrievedChunk]) -> list[dict]:
        return [
            {
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
            }
            for chunk in chunks
        ]

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        context_parts = []

        for chunk in chunks:
            context_parts.append(f"""
[Source: {chunk.document_name}, Page: {chunk.page_number}, Chunk: {chunk.chunk_id}, Score: {chunk.score:.4f}]
{chunk.text}
""".strip())

        return "\n\n".join(context_parts)

    def _retrieve_financial_highlight_candidates(
        self,
        candidate_top_k: int,
    ) -> list[RetrievedChunk]:
        chunk_lookup: dict[str, RetrievedChunk] = {}

        per_query_top_k = max(candidate_top_k, 10)
        chunks_per_metric = 2

        for queries in FINANCIAL_HIGHLIGHT_QUERIES.values():
            metric_chunk_lookup: dict[str, RetrievedChunk] = {}

            for query in queries:
                chunks = self.rag_pipeline.retriever.retrieve(
                    question=query,
                    top_k=per_query_top_k,
                )

                for chunk in chunks:
                    existing_chunk = metric_chunk_lookup.get(chunk.chunk_id)

                    if existing_chunk is None or chunk.score > existing_chunk.score:
                        metric_chunk_lookup[chunk.chunk_id] = chunk

            ranked_metric_chunks = sorted(
                metric_chunk_lookup.values(),
                key=lambda chunk: chunk.score,
                reverse=True,
            )

            selected_metric_chunks = ranked_metric_chunks[:chunks_per_metric]

            for chunk in selected_metric_chunks:
                existing_chunk = chunk_lookup.get(chunk.chunk_id)

                if existing_chunk is None or chunk.score > existing_chunk.score:
                    chunk_lookup[chunk.chunk_id] = chunk

        ranked_chunks = sorted(
            chunk_lookup.values(),
            key=lambda chunk: chunk.score,
            reverse=True,
        )

        return ranked_chunks

    def _build_extraction_prompt(self, context: str) -> str:
        return f"""
Retrieved context:
{context}

Task:
{EXTRACTION_QUESTION}

Return JSON in exactly this schema:

{{
  "document_name": "string or null",
  "fiscal_year": "string or null",
  "currency": "string or null",
  "financial_highlights": {{
    "consolidated_revenue_from_operations": {{
      "value": "string or null",
      "source_pages": [1, 2]
    }},
    "standalone_revenue_from_operations": {{
      "value": "string or null",
      "source_pages": [1, 2]
    }},
    "year_on_year_consolidated_revenue_growth": {{
      "value": "string or null",
      "source_pages": [1, 2]
    }},
    "total_dividend_per_share": {{
      "value": "string or null",
      "source_pages": [1, 2]
    }},
    "buyback_amount": {{
      "value": "string or null",
      "source_pages": [1, 2]
    }},
    "buyback_price_per_share": {{
      "value": "string or null",
      "source_pages": [1, 2]
    }}
  }}
}}
""".strip()

    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        cleaned_text = response_text.strip()

        fenced_match = re.search(
            r"```(?:json)?\s*(.*?)```",
            cleaned_text,
            re.DOTALL,
        )

        if fenced_match:
            cleaned_text = fenced_match.group(1).strip()

        start_index = cleaned_text.find("{")
        end_index = cleaned_text.rfind("}")

        if start_index == -1 or end_index == -1:
            raise ValueError("No JSON object found in LLM response.")

        json_text = cleaned_text[start_index : end_index + 1]

        return json.loads(json_text)

    def _empty_metric(self) -> dict:
        return {
            "value": None,
            "source_pages": [],
        }

    def _normalize_source_pages(self, pages: Any) -> list[int]:
        if not isinstance(pages, list):
            return []

        normalized_pages = []

        for page in pages:
            try:
                normalized_pages.append(int(page))
            except (TypeError, ValueError):
                continue

        return sorted(set(normalized_pages))

    def _normalize_metric(self, metric: Any) -> dict:
        if not isinstance(metric, dict):
            return self._empty_metric()

        value = self._normalize_metric_value(metric.get("value"))
        source_pages = self._normalize_source_pages(metric.get("source_pages", []))

        return {
            "value": value,
            "source_pages": source_pages,
        }

    def _normalize_extraction(
        self,
        extraction: dict[str, Any],
        chunks: list[RetrievedChunk],
    ) -> dict:
        financial_highlights = extraction.get("financial_highlights", {})

        if not isinstance(financial_highlights, dict):
            financial_highlights = {}

        normalized_highlights = {}

        for key in FINANCIAL_HIGHLIGHT_KEYS:
            normalized_highlights[key] = self._normalize_metric(
                financial_highlights.get(key)
            )

        document_name = extraction.get("document_name")

        if not document_name and chunks:
            document_name = chunks[0].document_name

        currency = extraction.get("currency")

        if currency is not None:
            currency_text = str(currency).strip().lower()

            if currency_text in ["`", "₹", "rs", "rs.", "rupee", "rupees", "inr"]:
                currency = "INR"

        return {
            "document_name": document_name,
            "fiscal_year": extraction.get("fiscal_year"),
            "currency": currency,
            "financial_highlights": normalized_highlights,
        }

    def _has_any_extracted_value(self, extraction: dict) -> bool:
        highlights = extraction.get("financial_highlights", {})

        return any(metric.get("value") is not None for metric in highlights.values())

    def extract_financial_highlights(
        self,
        top_k: int = 12,
    ) -> dict:
        total_start = time.perf_counter()

        retrieval_ms = 0.0
        rerank_ms = 0.0
        llm_ms = 0.0

        candidate_top_k = max(RERANK_TOP_K, top_k)

        retrieval_start = time.perf_counter()

        candidate_chunks = self._retrieve_financial_highlight_candidates(
            candidate_top_k=candidate_top_k,
        )

        retrieval_ms = self._elapsed_ms(retrieval_start)

        if not candidate_chunks:
            total_ms = self._elapsed_ms(total_start)

            timings_ms = self._build_timings(
                retrieval_ms=retrieval_ms,
                rerank_ms=rerank_ms,
                llm_ms=llm_ms,
                total_ms=total_ms,
            )

            return {
                "document_name": None,
                "fiscal_year": None,
                "currency": None,
                "financial_highlights": {
                    key: self._empty_metric() for key in FINANCIAL_HIGHLIGHT_KEYS
                },
                "sources": [],
                "metadata": {
                    "extraction_mode": "financial_highlights_json",
                    "retrieval_strategy": EXTRACTION_RETRIEVAL_STRATEGY,
                    "llm_provider": LLM_PROVIDER,
                    "is_answer_found": False,
                    "fallback_reason": "no_chunks_retrieved",
                    "sources_used": 0,
                    "timings_ms": timings_ms,
                    "extracted_field_count": 0,
                    "total_field_count": len(FINANCIAL_HIGHLIGHT_KEYS),
                    "extracted_fields": [],
                    "missing_fields": FINANCIAL_HIGHLIGHT_KEYS,
                },
            }

        # Structured extraction needs coverage across multiple fields.
        # We avoid one global rerank here because it can over-focus on one topic
        # such as dividends/buyback and drop revenue chunks.
        final_chunks = candidate_chunks[: max(top_k, 12)]

        context = self._format_context(final_chunks)
        extraction_prompt = self._build_extraction_prompt(context)

        llm_start = time.perf_counter()

        response_text = self.rag_pipeline.llm_client.generate_answer(
            user_prompt=extraction_prompt,
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
        )

        llm_ms = self._elapsed_ms(llm_start)

        total_ms = self._elapsed_ms(total_start)

        timings_ms = self._build_timings(
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
        )

        try:
            parsed_response = self._parse_json_response(response_text)
            normalized_extraction = self._normalize_extraction(
                extraction=parsed_response,
                chunks=final_chunks,
            )

            is_answer_found = self._has_any_extracted_value(normalized_extraction)
            extraction_quality = self._build_extraction_quality(normalized_extraction)

            fallback_reason = None if is_answer_found else "no_values_extracted"

        except Exception:
            normalized_extraction = {
                "document_name": (
                    final_chunks[0].document_name if final_chunks else None
                ),
                "fiscal_year": None,
                "currency": None,
                "financial_highlights": {
                    key: self._empty_metric() for key in FINANCIAL_HIGHLIGHT_KEYS
                },
            }

            is_answer_found = False
            fallback_reason = "json_parse_failed"
            extraction_quality = self._build_extraction_quality(normalized_extraction)

        return {
            **normalized_extraction,
            "sources": self._build_sources(final_chunks),
            "metadata": {
                "extraction_mode": "financial_highlights_json",
                "retrieval_strategy": EXTRACTION_RETRIEVAL_STRATEGY,
                "llm_provider": LLM_PROVIDER,
                "is_answer_found": is_answer_found,
                "fallback_reason": fallback_reason,
                "sources_used": len(final_chunks),
                **extraction_quality,
                "timings_ms": timings_ms,
            },
        }
