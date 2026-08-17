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
from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from index import hybrid_search, text_search, vector_search

load_dotenv()

GROUND_TRUTH_PATH = "data/retrieval_ground_truth.csv"
DETAILS_OUTPUT_PATH = "data/retrieval_evaluation_details.csv"
K_VALUES = [1, 3, 5, 10]

SearchFunction = Callable[..., list[dict[str, Any]]]


def make_chunk_id(doc_id: str, chunk_index: int | str) -> str:
    return f"{doc_id}:{int(chunk_index)}"


def split_chunk_id(chunk_id: str) -> tuple[str, int]:
    doc_id, chunk_index = chunk_id.rsplit(":", 1)
    return doc_id, int(chunk_index)


def load_ground_truth(path: str) -> list[dict[str, Any]]:
    """Load expected exact chunks and derive expected transcript IDs."""
    records = []
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            expected_chunk_ids = {
                value.strip()
                for value in row["expected_chunk_ids"].split("|")
                if value.strip()
            }
            expected_doc_ids = {
                split_chunk_id(chunk_id)[0]
                for chunk_id in expected_chunk_ids
            }
            record = {
                "query": row["query"],
                "expected_doc_ids": expected_doc_ids,
                "expected_chunk_ids": expected_chunk_ids,
            }
            if row.get("category"):
                record["category"] = row["category"]
            records.append(record)
    return records



def result_doc_id(result: dict[str, Any]) -> str:
    return str(result["doc_id"])


def result_chunk_id(result: dict[str, Any]) -> str:
    if result.get("chunk_id"):
        return str(result["chunk_id"])
    return make_chunk_id(result["doc_id"], result["chunk_index"])


def retrieve(
    search_function: SearchFunction,
    query: str,
    k: int,
) -> list[dict[str, Any]]:
    return search_function(query=query, limit=k)


def hit_at_k(
    results: list[dict[str, Any]],
    expected_doc_ids: set[str],
) -> int:
    return int(any(
        result_doc_id(result) in expected_doc_ids
        for result in results
    ))


def exact_chunk_hit_at_k(
    results: list[dict[str, Any]],
    expected_chunk_ids: set[str],
) -> int:
    return int(any(
        result_chunk_id(result) in expected_chunk_ids
        for result in results
    ))


def reciprocal_rank(
    results: list[dict[str, Any]],
    expected_doc_ids: set[str],
) -> float:
    for rank, result in enumerate(results, start=1):
        if result_doc_id(result) in expected_doc_ids:
            return 1.0 / rank
    return 0.0


def exact_chunk_reciprocal_rank(
    results: list[dict[str, Any]],
    expected_chunk_ids: set[str],
) -> float:
    for rank, result in enumerate(results, start=1):
        if result_chunk_id(result) in expected_chunk_ids:
            return 1.0 / rank
    return 0.0


def evaluate_search_function(
    search_function: SearchFunction,
    ground_truth: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    details = []

    for record in ground_truth:
        results = retrieve(search_function, record["query"], k)
        expected_doc_ids = record["expected_doc_ids"]
        expected_chunk_ids = record["expected_chunk_ids"]

        details.append({
            "query": record["query"],
            "category": record.get("category", ""),
            "expected_doc_ids": "|".join(sorted(expected_doc_ids)),
            "expected_chunk_ids": "|".join(sorted(expected_chunk_ids)),
            "retrieved_doc_ids": "|".join(
                result_doc_id(result) for result in results
            ),
            "retrieved_chunk_ids": "|".join(
                result_chunk_id(result) for result in results
            ),
            "doc_hit": hit_at_k(results, expected_doc_ids),
            "chunk_hit": exact_chunk_hit_at_k(results, expected_chunk_ids),
            "doc_reciprocal_rank": reciprocal_rank(results, expected_doc_ids),
            "chunk_reciprocal_rank": exact_chunk_reciprocal_rank(
                results, expected_chunk_ids
            ),
        })

    query_count = len(details)
    return {
        "queries": query_count,
        "k": k,
        "doc_hit_rate": (
            sum(row["doc_hit"] for row in details) / query_count
            if query_count else 0.0
        ),
        "chunk_hit_rate": (
            sum(row["chunk_hit"] for row in details) / query_count
            if query_count else 0.0
        ),
        "doc_mrr": (
            sum(row["doc_reciprocal_rank"] for row in details) / query_count
            if query_count else 0.0
        ),
        "chunk_mrr": (
            sum(row["chunk_reciprocal_rank"] for row in details) / query_count
            if query_count else 0.0
        ),
        "details": details,
    }


def save_details(details: list[dict[str, Any]], path: str) -> None:
    fieldnames = [
        "method",
        "k",
        "query",
        "category",
        "expected_doc_ids",
        "expected_chunk_ids",
        "retrieved_doc_ids",
        "retrieved_chunk_ids",
        "doc_hit",
        "chunk_hit",
        "doc_reciprocal_rank",
        "chunk_reciprocal_rank",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)


def print_failures(evaluation: dict[str, Any]) -> None:
    for row in evaluation["details"]:
        if not row["doc_hit"] or not row["chunk_hit"]:
            print(f"Query: {row['query']}")
            print(f"Expected documents: {row['expected_doc_ids']}")
            print(f"Retrieved documents: {row['retrieved_doc_ids']}")
            print(f"Expected chunks: {row['expected_chunk_ids']}")
            print(f"Retrieved chunks: {row['retrieved_chunk_ids']}")
            print()


def main() -> None:
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    ground_truth = [
        record for record in ground_truth
        if "NEEDS_CONTEXT" not in record["expected_chunk_ids"]
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
                f"doc_hit_rate={evaluation['doc_hit_rate']:.3f}, "
                f"chunk_hit_rate={evaluation['chunk_hit_rate']:.3f}, "
                f"doc_mrr={evaluation['doc_mrr']:.3f}, "
                f"chunk_mrr={evaluation['chunk_mrr']:.3f}"
            )
            all_details.extend(
                {
                    "method": method_name,
                    "k": k,
                    **detail,
                }
                for detail in evaluation["details"]
            )

    save_details(all_details, DETAILS_OUTPUT_PATH)
    print(f"\nDetailed results saved to: {DETAILS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()