# scripts/evaluate_rag_answers.py

import json
from pathlib import Path

from src.finsight.config import DEFAULT_TOP_K
from src.finsight.rag.pipeline import FALLBACK_ANSWER, RAGPipeline

EVAL_FILE_PATH = Path("data/evaluation/rag_answer_eval.json")

FORBIDDEN_ANSWER_PATTERNS = [
    "Source:",
    "Page:",
    "Chunk:",
    "infosys_annual_report.pdf",
]


def load_eval_items() -> list[dict]:
    if not EVAL_FILE_PATH.exists():
        raise FileNotFoundError(f"Evaluation file not found: {EVAL_FILE_PATH}")

    with open(EVAL_FILE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def contains_expected_terms(answer: str, expected_terms: list[str]) -> bool:
    answer_lower = answer.lower()

    return all(term.lower() in answer_lower for term in expected_terms)


def contains_forbidden_patterns(answer: str) -> bool:
    return any(pattern in answer for pattern in FORBIDDEN_ANSWER_PATTERNS)


def has_expected_source_page(
    sources: list[dict],
    expected_pages: list[int],
) -> bool:
    if not expected_pages:
        return len(sources) == 0

    source_pages = {source["page_number"] for source in sources}

    return bool(source_pages.intersection(set(expected_pages)))


def evaluate_rag_answers(top_k: int = DEFAULT_TOP_K) -> None:
    eval_items = load_eval_items()
    pipeline = RAGPipeline()

    total = 0
    passed = 0

    answerable_total = 0
    answerable_passed = 0

    unanswerable_total = 0
    unanswerable_passed = 0

    print("\nFull RAG Answer Evaluation")
    print("=" * 100)

    for idx, item in enumerate(eval_items, start=1):
        question = item["question"]
        answer_type = item["answer_type"]
        expected_terms = item["expected_answer_contains"]
        expected_source_pages = item["expected_source_pages"]

        result = pipeline.ask(
            question=question,
            top_k=top_k,
        )

        answer = result["answer"]
        sources = result["sources"]
        metadata = result["metadata"]

        total += 1

        checks = {}

        checks["expected_terms_present"] = contains_expected_terms(
            answer=answer,
            expected_terms=expected_terms,
        )

        checks["no_inline_source_citations"] = not contains_forbidden_patterns(
            answer=answer,
        )

        checks["expected_source_page_present"] = has_expected_source_page(
            sources=sources,
            expected_pages=expected_source_pages,
        )

        if answer_type == "answerable":
            answerable_total += 1

            checks["is_answer_found_true"] = metadata["is_answer_found"] is True

            checks["sources_present"] = len(sources) > 0

        else:
            unanswerable_total += 1

            checks["is_answer_found_false"] = metadata["is_answer_found"] is False

            checks["fallback_answer_exact"] = answer == FALLBACK_ANSWER
            checks["sources_empty"] = len(sources) == 0

        item_passed = all(checks.values())

        if item_passed:
            passed += 1

            if answer_type == "answerable":
                answerable_passed += 1
            else:
                unanswerable_passed += 1

        status = "PASS" if item_passed else "FAIL"

        print(f"\n{idx}. {question}")
        print(f"Answer type: {answer_type}")
        print(f"Answer: {answer}")
        print(f"Source pages: {[source['page_number'] for source in sources]}")
        print(f"Metadata is_answer_found: {metadata['is_answer_found']}")
        print(f"Fallback reason: {metadata['fallback_reason']}")
        print("Checks:")

        for check_name, check_value in checks.items():
            check_status = "PASS" if check_value else "FAIL"
            print(f"  - {check_name}: {check_status}")

        print(f"Status: {status}")

    pass_rate = passed / total if total else 0.0
    answerable_pass_rate = (
        answerable_passed / answerable_total if answerable_total else 0.0
    )
    unanswerable_pass_rate = (
        unanswerable_passed / unanswerable_total if unanswerable_total else 0.0
    )

    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)
    print(f"Total questions: {total}")
    print(f"Passed: {passed}")
    print(f"Pass rate: {pass_rate:.2%}")
    print(f"Answerable questions: {answerable_total}")
    print(f"Answerable passed: {answerable_passed}")
    print(f"Answerable pass rate: {answerable_pass_rate:.2%}")
    print(f"Unanswerable questions: {unanswerable_total}")
    print(f"Unanswerable passed: {unanswerable_passed}")
    print(f"Unanswerable pass rate: {unanswerable_pass_rate:.2%}")


if __name__ == "__main__":
    evaluate_rag_answers()
