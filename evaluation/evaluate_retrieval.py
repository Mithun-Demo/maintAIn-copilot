
import json
import re
from pathlib import Path

from src.retrieval import search_historical_failures
from src.vector_retrieval import build_vector_store


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULT_FILE = (
    PROJECT_DIR
    / "evaluation"
    / "retrieval_evaluation_results.json"
)

# These labels are manually assigned using our synthetic demo records.
# The queries use alternative wording to test retrieval robustness.
TEST_CASES = [
    {
        "query": (
            "The pump is shaking and making an unfamiliar "
            "sound while getting hotter."
        ),
        "expected_ids": ["HF-001"],
    },
    {
        "query": (
            "Could poor lubrication explain rising bearing "
            "heat and vibration?"
        ),
        "expected_ids": ["HF-001"],
    },
    {
        "query": (
            "Flow from the pump has fallen and the outlet "
            "gauge pressure is low."
        ),
        "expected_ids": ["HF-002"],
    },
    {
        "query": (
            "The discharge side seems restricted and the pump "
            "is delivering less fluid than usual."
        ),
        "expected_ids": ["HF-002"],
    },
    {
        "query": (
            "Rotating shaft oscillation suggests an alignment "
            "or balance problem."
        ),
        "expected_ids": ["HF-003"],
    },
    {
        "query": (
            "Check whether coupling alignment is causing noisy "
            "vibration in rotating machinery."
        ),
        "expected_ids": ["HF-003"],
    },
    {
        "query": (
            "Repeated equipment alerts show both overheating "
            "and vibration; sensor readings may be inaccurate."
        ),
        "expected_ids": ["HF-004"],
    },
    {
        "query": (
            "After multiple maintenance warnings, verify "
            "instrumentation and possible operating problems."
        ),
        "expected_ids": ["HF-004"],
    },
]


def extract_case_ids(text):
    """Extract unique case IDs while preserving their order."""
    return list(
        dict.fromkeys(
            re.findall(r"\bHF-\d+\b", str(text))
        )
    )


def relevant_rank(retrieved_ids, expected_ids):
    """
    Return the rank of the first expected relevant case.
    Return None if no expected case appears in the top results.
    """
    for rank, case_id in enumerate(retrieved_ids, start=1):
        if case_id in expected_ids:
            return rank
    return None


def main():
    print("Building the vector index once...")
    vector_store = build_vector_store()

    results = []

    for index, test in enumerate(TEST_CASES, start=1):
        query = test["query"]
        expected_ids = test["expected_ids"]

        print(f"\nTest {index}/{len(TEST_CASES)}: {query}")

        # Keyword retrieval
        try:
            keyword_result = search_historical_failures.invoke(
                {"query": query}
            )
            keyword_ids = extract_case_ids(keyword_result)[:3]
            keyword_error = None
        except Exception as exc:
            keyword_ids = []
            keyword_error = f"{type(exc).__name__}: {exc}"

        # Vector retrieval
        try:
            vector_documents = vector_store.similarity_search(
                query, k=3
            )
            vector_ids = [
                document.metadata["case_id"]
                for document in vector_documents
                if "case_id" in document.metadata
            ]
            vector_error = None
        except Exception as exc:
            vector_ids = []
            vector_error = f"{type(exc).__name__}: {exc}"

        keyword_rank = relevant_rank(keyword_ids, expected_ids)
        vector_rank = relevant_rank(vector_ids, expected_ids)

        entry = {
            "query": query,
            "expected_ids": expected_ids,
            "keyword_top3": keyword_ids,
            "keyword_relevant_rank": keyword_rank,
            "keyword_error": keyword_error,
            "vector_top3": vector_ids,
            "vector_relevant_rank": vector_rank,
            "vector_error": vector_error,
        }
        results.append(entry)

        print(f"Expected: {expected_ids}")
        print(
            f"Keyword top 3: {keyword_ids} | "
            f"Relevant rank: {keyword_rank}"
        )
        print(
            f"Vector top 3:  {vector_ids} | "
            f"Relevant rank: {vector_rank}"
        )

    total = len(results)

    keyword_hit1 = sum(
        row["keyword_relevant_rank"] == 1 for row in results
    )
    vector_hit1 = sum(
        row["vector_relevant_rank"] == 1 for row in results
    )

    keyword_hit3 = sum(
        row["keyword_relevant_rank"] is not None for row in results
    )
    vector_hit3 = sum(
        row["vector_relevant_rank"] is not None for row in results
    )

    keyword_mrr = sum(
        1 / row["keyword_relevant_rank"]
        if row["keyword_relevant_rank"] is not None
        else 0
        for row in results
    ) / total

    vector_mrr = sum(
        1 / row["vector_relevant_rank"]
        if row["vector_relevant_rank"] is not None
        else 0
        for row in results
    ) / total

    summary = {
        "dataset": "synthetic_demo_records",
        "queries_evaluated": total,
        "keyword_hit_at_1": round(keyword_hit1 / total, 4),
        "vector_hit_at_1": round(vector_hit1 / total, 4),
        "keyword_hit_at_3": round(keyword_hit3 / total, 4),
        "vector_hit_at_3": round(vector_hit3 / total, 4),
        "keyword_mrr_at_3": round(keyword_mrr, 4),
        "vector_mrr_at_3": round(vector_mrr, 4),
        "results": results,
        "limitation": (
            "Small synthetic dataset with manually labeled queries. "
            "Results do not establish real-world industrial accuracy."
        ),
    }

    RESULT_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== RETRIEVAL EVALUATION SUMMARY ===")
    print(f"Queries evaluated: {total}")
    print(f"Keyword Hit@1: {keyword_hit1}/{total} "
          f"({keyword_hit1 / total * 100:.1f}%)")
    print(f"Vector Hit@1:  {vector_hit1}/{total} "
          f"({vector_hit1 / total * 100:.1f}%)")
    print(f"Keyword Hit@3: {keyword_hit3}/{total} "
          f"({keyword_hit3 / total * 100:.1f}%)")
    print(f"Vector Hit@3:  {vector_hit3}/{total} "
          f"({vector_hit3 / total * 100:.1f}%)")
    print(f"Keyword MRR@3: {keyword_mrr:.3f}")
    print(f"Vector MRR@3:  {vector_mrr:.3f}")
    print(f"\nDetailed results saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()
