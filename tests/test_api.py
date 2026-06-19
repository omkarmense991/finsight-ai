# tests/test_api.py

from fastapi.testclient import TestClient

from src.finsight.api.main import app

client = TestClient(app)


class MockRAGPipeline:
    def ask(self, question: str, top_k: int = 5) -> dict:
        return {
            "question": question,
            "answer": "Mock answer from uploaded documents.",
            "sources": [
                {
                    "document_name": "mock_report.pdf",
                    "page_number": 1,
                    "chunk_id": "mock_report.pdf_p1_c1",
                    "score": 0.95,
                }
            ],
            "metadata": {
                "top_k": top_k,
                "min_score": 0.025,
                "best_score": 0.95,
                "best_retrieval_score": 0.95,
                "best_final_score": 0.95,
                "retrieved_chunks": 1,
                "retrieval_strategy": "hybrid_faiss_bm25_rrf_blended_rerank",
                "llm_provider": "gemini",
                "is_answer_found": True,
                "fallback_reason": None,
                "timings_ms": {
                    "retrieval_ms": 10.0,
                    "rerank_ms": 5.0,
                    "llm_ms": 100.0,
                    "total_ms": 115.0,
                },
            },
        }


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "FinSight AI API"


def test_ask_question_success():
    app.state.rag_pipeline = MockRAGPipeline()

    response = client.post(
        "/ask",
        json={
            "question": "What dividend did Infosys announce?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What dividend did Infosys announce?"
    assert data["answer"] == "Mock answer from uploaded documents."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["document_name"] == "mock_report.pdf"
    assert (
        data["metadata"]["retrieval_strategy"] == "hybrid_faiss_bm25_rrf_blended_rerank"
    )
    assert data["metadata"]["is_answer_found"] is True
    assert data["metadata"]["timings_ms"]["total_ms"] == 115.0


def test_ask_question_validation_error_for_short_question():
    response = client.post(
        "/ask",
        json={
            "question": "Hi",
            "top_k": 5,
        },
    )

    assert response.status_code == 422


def test_ask_question_validation_error_for_invalid_top_k():
    response = client.post(
        "/ask",
        json={
            "question": "What dividend did Infosys announce?",
            "top_k": 50,
        },
    )

    assert response.status_code == 422


def test_upload_rejects_non_pdf_file():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "notes.txt",
                b"This is not a PDF file.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported."
