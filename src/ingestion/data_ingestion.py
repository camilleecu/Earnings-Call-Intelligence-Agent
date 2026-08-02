"""
loading data and writing to Postgres
data_ingestion.py = fetch, normalize, chunk, embed, store.
"""

import os
import requests
from typing import List, Dict, Any, Optional
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


BASE_URL = "https://api.roic.ai/v2/company/earnings-calls/latest"
API_KEY = os.environ["ROIC_API_KEY"]
conn_str = os.getenv("PG_CONN_STR")


def fetch_data(ticker: str, format: str = "json") -> Dict[str, Any]:
    """
    Fetch the latest earnings call transcript for a given ticker from ROIC.ai.
    
    Returns a dict like:
    {
      "symbol": "AAPL",
      "year": 2024,
      "quarter": 4,
      "date": "2024-10-31",
      "content": "..."
    }
    """
    url = f"{BASE_URL}/{ticker}"
    params = {
        "apikey": API_KEY,
        "format": format,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def normalize_example(example: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = f"{example.get('symbol')}_{example.get('year')}Q{example.get('quarter')}"

    return {
        "doc_id": doc_id,
        "text": example["content"],  # ROIC.ai transcript text
        "metadata": {
            "source": "roic.ai",
            "symbol": example.get("symbol"),
            "year": example.get("year"),
            "quarter": example.get("quarter"),
            "date": example.get("date"),
        },
    }




def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks



model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    # You can add batch_size here if needed
    vectors = model.encode(chunks, convert_to_tensor=True, batch_size=64)
    return vectors.tolist()




def load_to_db(
    conn_str: str,
    normalized: Dict[str, Any],
    chunks: List[str],
    embeddings: List[List[float]],
) -> None:
    """
    Insert one transcript's chunks into the earnings_chunks table.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    with psycopg.connect(conn_str) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                cur.execute(
                    """
                    INSERT INTO earnings_chunks (
                        doc_id, chunk_index, text, embedding,
                        symbol, year, quarter, date, source
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        normalized["doc_id"],
                        i,
                        chunk,
                        emb,
                        normalized["metadata"]["symbol"],
                        normalized["metadata"]["year"],
                        normalized["metadata"]["quarter"],
                        normalized["metadata"]["date"],
                        normalized["metadata"]["source"],
                    ),
                )



def ingest_ticker(ticker: str) -> None:
    raw_transcript = fetch_data(ticker)
    normalized_transcript = normalize_example(raw_transcript)
    chunks = chunk_text(normalized_transcript["text"], chunk_size=1000, overlap=150)
    embeddings = embed_chunks(chunks)
    load_to_db(conn_str, normalized_transcript, chunks, embeddings)



def main() -> None:
    ticker = os.getenv("INGEST_TICKER", "AAPL")
    ingest_ticker(ticker)
    # build_indexes()
    # save_eval_set(normalized_examples)


if __name__ == "__main__":
    main()