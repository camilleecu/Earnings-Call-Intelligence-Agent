"""
Evaluate text, vector, and hybrid retrieval for the Earnings Call RAG Assistant.

The script compares retrieval methods against a labeled ground-truth CSV.

Metrics:
- Hit Rate@k: whether an expected transcript appears in the top-k results.
- MRR: how highly the first expected transcript is ranked.

Input:
- data/retrieval_ground_truth.csv

Output:
- data/retrieval_evaluation_details.csv
"""

import csv
from typing import Any, Callable

from dotenv import load_dotenv

from index import hybrid_search, text_search, vector_search

load_dotenv()

GROUND_TRUTH_PATH = "data/retrieval_ground_truth.csv"
DETAILS_OUTPUT_PATH = "data/retrieval_evaluation_details.csv"
K_VALUES = [1, 3, 5, 10]

SearchFunction = Callable[[str, int], list[dict[str, Any]]]


def load_ground_truth(path: str) -> list[dict[str, Any]]:
    """
    Load retrieval questions and expected transcript IDs from a CSV file.

    The CSV must contain:
    - query
    - expected_doc_ids

    Multiple acceptable transcript IDs can be separated with `|`.
    """
    records = []

    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            expected_doc_ids = {
                value.strip()
                for value in row["expected_doc_ids"].split("|")
                if value.strip()
            }

            records.append(
                {
                    "query": row["query"],
                    "expected_doc_ids": expected_doc_ids,
                }
            )

    return records


def result_doc_id(result: dict[str, Any]) -> str | None:
    """
    Return the transcript-level ID from one retrieved chunk.

    Each transcript is divided into chunks, but all chunks from the same
    transcript share the same `doc_id`.
    """
    return result.get("doc_id")


def retrieve(
    search_function: SearchFunction,
    query: str,
    k: int,
) -> list[dict[str, Any]]:
    """
    Run one search function and return its top-k results.
    """
    return search_function(query=query, limit=k)


def hit_at_k(
    results: list[dict[str, Any]],
    expected_doc_ids: set[str],
) -> int:
    """
    Return 1 if any top-k result belongs to an expected transcript.

    Return 0 when none of the retrieved chunks belongs to an expected
    transcript.
    """
    return int(
        any(
            result_doc_id(result) in expected_doc_ids
            for result in results
        )
    )


def reciprocal_rank(
    results: list[dict[str, Any]],
    expected_doc_ids: set[str],
) -> float:
    """
    Return the reciprocal rank of the first relevant result.

    Rank 1 returns 1.0, rank 2 returns 0.5, rank 3 returns 0.333,
    and no relevant result returns 0.0.
    """
    for rank, result in enumerate(results, start=1):
        if result_doc_id(result) in expected_doc_ids:
            return 1.0 / rank

    return 0.0


def evaluate_search_function(
    search_function: SearchFunction,
    ground_truth: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    """
    Evaluate one search function across all ground-truth questions.

    Returns overall Hit Rate@k, MRR@k, and query-level details for
    investigating retrieval failures.
    """
    hits = []
    reciprocal_ranks = []
    details = []

    for record in ground_truth:
        results = retrieve(search_function, record["query"], k)

        hit = hit_at_k(
            results,
            record["expected_doc_ids"],
        )

        rank_score = reciprocal_rank(
            results,
            record["expected_doc_ids"],
        )

        hits.append(hit)
        reciprocal_ranks.append(rank_score)

        details.append(
            {
                "query": record["query"],
                "expected_doc_ids": "|".join(
                    sorted(record["expected_doc_ids"])
                ),
                "retrieved_doc_ids": "|".join(
                    result_doc_id(result) or ""
                    for result in results
                ),
                "hit": hit,
                "reciprocal_rank": rank_score,
            }
        )

    query_count = len(ground_truth)

    return {
        "hit_rate": sum(hits) / query_count if query_count else 0.0,
        "mrr": (
            sum(reciprocal_ranks) / query_count
            if query_count
            else 0.0
        ),
        "details": details,
    }


def save_details(
    details: list[dict[str, Any]],
    path: str,
) -> None:
    """
    Save query-level evaluation results to a CSV file.

    The output can be filtered by `hit == 0` to inspect failed queries.
    """
    fieldnames = [
        "method",
        "k",
        "query",
        "expected_doc_ids",
        "retrieved_doc_ids",
        "hit",
        "reciprocal_rank",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)


def main() -> None:
    """
    Evaluate text, vector, and hybrid retrieval at several k values.

    The ambiguous `NEEDS_CONTEXT` query is excluded because it represents
    a clarification case rather than a normal transcript-retrieval case.
    """
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    ground_truth = [
        record
        for record in ground_truth
        if "NEEDS_CONTEXT" not in record["expected_doc_ids"]
    ]

    search_methods = {
        "text": text_search,
        "vector": vector_search,
        "hybrid": hybrid_search,
    }

    all_details = []

    for method_name, search_function in search_methods.items():
        print(f"\n=== {method_name.upper()} ===")

        for k in K_VALUES:
            evaluation = evaluate_search_function(
                search_function=search_function,
                ground_truth=ground_truth,
                k=k,
            )

            print(
                f"k={k}: "
                f"hit_rate={evaluation['hit_rate']:.3f}, "
                f"mrr={evaluation['mrr']:.3f}"
            )

            for detail in evaluation["details"]:
                all_details.append(
                    {
                        "method": method_name,
                        "k": k,
                        **detail,
                    }
                )

    save_details(all_details, DETAILS_OUTPUT_PATH)

    print(
        f"\nDetailed results saved to: "
        f"{DETAILS_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()