# scripts/evaluate_retrieval.py

import argparse
import json
from pathlib import Path

from src.finsight.config import DEFAULT_TOP_K, RERANK_TOP_K
from src.finsight.rag.reranker import CrossEncoderReranker
from src.finsight.rag.retriever import MultiQueryRetriever

EVAL_FILE_PATH = Path("data/evaluation/rag_eval_questions.json")


def load_eval_questions() -> list[dict]:
    if not EVAL_FILE_PATH.exists():
        raise FileNotFoundError(f"Evaluation file not found: {EVAL_FILE_PATH}")

    with open(EVAL_FILE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def find_first_relevant_rank(
    retrieved_pages: list[int],
    expected_pages: set[int],
) -> int | None:
    for rank, page_number in enumerate(retrieved_pages, start=1):
        if page_number in expected_pages:
            return rank

    return None


def evaluate_retrieval(
    top_k: int = DEFAULT_TOP_K,
    use_reranker: bool = False,
    candidate_k: int = RERANK_TOP_K,
) -> None:
    eval_questions = load_eval_questions()
    retriever = MultiQueryRetriever()
    reranker = CrossEncoderReranker() if use_reranker else None

    total_answerable = 0
    answerable_hits = 0
    reciprocal_rank_sum = 0.0

    total_unanswerable = 0

    retrieval_mode = (
        f"hybrid_top_{candidate_k}_blended_rerank_top_{top_k}"
        if use_reranker
        else f"hybrid_top_{top_k}"
    )

    print("\nRetrieval Evaluation")
    print("=" * 100)
    print(f"Retrieval mode: {retrieval_mode}")

    for idx, item in enumerate(eval_questions, start=1):
        question = item["question"]
        expected_pages = set(item["expected_pages"])
        answer_type = item["answer_type"]

        if use_reranker:
            candidate_chunks = retriever.retrieve(
                question=question,
                top_k=candidate_k,
            )

            retrieved_chunks = reranker.rerank(
                question=question,
                chunks=candidate_chunks,
                top_k=top_k,
            )
        else:
            retrieved_chunks = retriever.retrieve(
                question=question,
                top_k=top_k,
            )

        retrieved_pages = [chunk.page_number for chunk in retrieved_chunks]

        if answer_type == "answerable":
            total_answerable += 1

            first_relevant_rank = find_first_relevant_rank(
                retrieved_pages=retrieved_pages,
                expected_pages=expected_pages,
            )

            is_hit = first_relevant_rank is not None

            if is_hit:
                answerable_hits += 1
                reciprocal_rank = 1 / first_relevant_rank
                reciprocal_rank_sum += reciprocal_rank
                status = "PASS"
            else:
                reciprocal_rank = 0.0
                status = "FAIL"

        else:
            total_unanswerable += 1
            first_relevant_rank = None
            reciprocal_rank = None
            status = "INFO"

        print(f"\n{idx}. {question}")
        print(f"Answer type: {answer_type}")
        print(f"Expected pages: {sorted(expected_pages)}")
        print(f"Retrieved pages: {retrieved_pages}")
        print(f"First relevant rank: {first_relevant_rank}")
        print(f"Reciprocal rank: {reciprocal_rank}")
        print(f"Status: {status}")

    recall_at_k = answerable_hits / total_answerable if total_answerable else 0.0
    mrr_at_k = reciprocal_rank_sum / total_answerable if total_answerable else 0.0

    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)
    print(f"Retrieval mode: {retrieval_mode}")
    print(f"Top K: {top_k}")
    print(f"Answerable questions: {total_answerable}")
    print(f"Answerable hits: {answerable_hits}")
    print(f"Recall@{top_k}: {recall_at_k:.2%}")
    print(f"MRR@{top_k}: {mrr_at_k:.2%}")
    print(f"Unanswerable questions: {total_unanswerable}")
    print(
        "Note: Unanswerable questions are mainly evaluated through "
        "LLM fallback, not retrieval-only recall."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of final retrieved chunks to evaluate.",
    )

    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Whether to rerank retrieved candidates.",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=RERANK_TOP_K,
        help="Number of candidates to retrieve before reranking.",
    )

    args = parser.parse_args()

    evaluate_retrieval(
        top_k=args.top_k,
        use_reranker=args.use_reranker,
        candidate_k=args.candidate_k,
    )
