"""
index.py responsible for:

text search queries,

vector search queries,

hybrid search with RRF,

any helper functions for ranking/fusing results.
"""


import os
from typing import List, Dict, Any

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


def _embed_query(query: str):
    vec = _vector_model.encode(query, convert_to_tensor=True)
    return vec.tolist()


def normalize_for_text_search(query: str) -> str:
    stopwords = {
        "what", "did", "do", "does", "is", "are", "was", "were",
        "the", "a", "an", "to", "of", "and", "or", "about",
        "say", "said", "tell", "told", "me", "us", "you"
    }
    tokens = [t.lower().strip("?.!,") for t in query.split()]
    tokens = [t for t in tokens if t and t not in stopwords]
    return " | ".join(tokens)

def text_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    relaxed_query = normalize_for_text_search(query)

    if not relaxed_query:
        return []


    sql = """
        WITH q AS (
            SELECT to_tsquery('english', %s) AS query
        )
        SELECT
            id, doc_id, chunk_index, symbol, year, quarter, date, source, text,
            ts_rank(text_search, q.query) AS score
        FROM earnings_chunks, q
        WHERE text_search @@ q.query
        ORDER BY score DESC, id ASC
        LIMIT %s
    """

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (relaxed_query, limit))
            rows = cur.fetchall()
            cols = [d.name if hasattr(d, "name") else d[0] for d in cur.description]

    return [dict(zip(cols, row)) for row in rows]


def vector_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    query_vec = _embed_query(query)
    sql = """
        SELECT
            id, doc_id, chunk_index, symbol, year, quarter, date, source, text,
            1 - (embedding <=> %s::vector) AS score
        FROM earnings_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query_vec, query_vec, limit))
            rows = cur.fetchall()
            cols = [d.name if hasattr(d, "name") else d[0] for d in cur.description]

    return [dict(zip(cols, row)) for row in rows]


def rrf_fuse(results_list: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    fused: Dict[Any, Dict[str, Any]] = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            doc_key = doc.get("id", doc.get("doc_id"))
            if doc_key not in fused:
                fused[doc_key] = dict(doc)
                fused[doc_key]["rrf_score"] = 0.0
            fused[doc_key]["rrf_score"] += 1.0 / (k + rank)

    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)


def hybrid_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    text_results = text_search(query, limit=limit)
    vector_results = vector_search(query, limit=limit)
    fused = rrf_fuse([text_results, vector_results], k=60)
    return fused[:limit]


if __name__ == "__main__":
    q = os.getenv("TEST_QUERY", "What did Apple say about margins?")
    results = hybrid_search(q, limit=5)
    for r in results:
        print(r["symbol"], r["year"], r["quarter"], r.get("rrf_score"), r["text"][:200])