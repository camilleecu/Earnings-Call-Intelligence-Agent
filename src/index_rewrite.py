"""Hybrid retrieval for earnings-call transcript chunks."""

import os
from typing import Any, Dict, List, Optional

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

PG_CONN_STR = os.getenv("PG_CONN_STR")
if not PG_CONN_STR:
    raise RuntimeError("PG_CONN_STR is not set")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
_vector_model = SentenceTransformer(EMBED_MODEL_NAME)


def _connect():
    conn = psycopg.connect(PG_CONN_STR)
    register_vector(conn)
    return conn


def _embed_query(query: str) -> List[float]:
    return _vector_model.encode(query, convert_to_numpy=True).tolist()


def _to_dicts(cur, rows) -> List[Dict[str, Any]]:
    columns = [column.name for column in cur.description]
    return [dict(zip(columns, row)) for row in rows]


def normalize_for_text_search(query: str) -> str:
    stopwords = {
        "what", "did", "do", "does", "is", "are", "was", "were",
        "the", "a", "an", "to", "of", "and", "or", "about",
        "say", "said", "tell", "told", "me", "us", "you",
    }

    tokens = [token.lower().strip("?.!,") for token in query.split()]
    tokens = [token for token in tokens if token and token not in stopwords]

    return " or ".join(tokens)


def text_search(
    query: str,
    limit: int = 5,
    symbol: Optional[str] = None,
    year: Optional[int] = None,
    quarter: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Full-text search with optional transcript metadata filters."""
    query = normalize_for_text_search(query)
    print(f"Normalized query for text search: {query}")

    sql = """
    WITH q AS (
        SELECT websearch_to_tsquery('english', %s) AS query
    )
    SELECT
        id, doc_id, chunk_index, symbol, year, quarter, date, source, text,
        ts_rank_cd(text_search, q.query) AS score
    FROM earnings_chunks, q
    WHERE text_search @@ q.query
      AND (%s::text IS NULL OR symbol = %s::text)
      AND (%s::integer IS NULL OR year = %s::integer)
      AND (%s::integer IS NULL OR quarter = %s::integer)
    ORDER BY score DESC, id ASC
    LIMIT %s
    """

    params = (query, symbol, symbol, year, year, quarter, quarter, limit)

    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return _to_dicts(cur, cur.fetchall())


def vector_search(
    query: str,
    limit: int = 5,
    symbol: Optional[str] = None,
    year: Optional[int] = None,
    quarter: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Vector search with optional transcript metadata filters."""
    if not query.strip():
        return []

    query_vector = _embed_query(query)

    sql = """
    SELECT
        id, doc_id, chunk_index, symbol, year, quarter, date, source, text,
        1 - (embedding <=> %s::vector) AS score
    FROM earnings_chunks
    WHERE (%s::text IS NULL OR symbol = %s::text)
        AND (%s::integer IS NULL OR year = %s::integer)
        AND (%s::integer IS NULL OR quarter = %s::integer)
    ORDER BY embedding <=> %s::vector
    LIMIT %s
    """

    params = (
        query_vector,
        symbol, symbol,
        year, year,
        quarter, quarter,
        query_vector,
        limit,
    )

    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return _to_dicts(cur, cur.fetchall())


def rrf_fuse(
    results_list: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse text and vector rankings with reciprocal-rank fusion."""
    fused: Dict[Any, Dict[str, Any]] = {}

    for results in results_list:
        for rank, document in enumerate(results, start=1):
            document_id = document["id"]

            if document_id not in fused:
                fused[document_id] = dict(document)
                fused[document_id]["rrf_score"] = 0.0

            fused[document_id]["rrf_score"] += 1.0 / (k + rank)

    return sorted(fused.values(), key=lambda doc: doc["rrf_score"], reverse=True)


def hybrid_search(
    query: str,
    limit: int = 5,
    symbol: Optional[str] = None,
    year: Optional[int] = None,
    quarter: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run filtered text and vector retrieval, then fuse with RRF."""
    candidate_limit = limit * 3

    text_results = text_search(
        query=query,
        limit=candidate_limit,
        symbol=symbol,
        year=year,
        quarter=quarter,
    )
    vector_results = vector_search(
        query=query,
        limit=candidate_limit,
        symbol=symbol,
        year=year,
        quarter=quarter,
    )

    return rrf_fuse([text_results, vector_results])[:limit]


if __name__ == "__main__":
    query = os.getenv("TEST_QUERY", "What did Apple say about margins?")
    symbol = os.getenv("TEST_SYMBOL")

    results = hybrid_search(query=query, symbol=symbol, limit=5)

    for result in results:
        print(
            result["symbol"],
            result["year"],
            result["quarter"],
            result["text"][:200].replace("\n", " "),
        )
