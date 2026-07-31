import os
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.roic.ai/v2/company/earnings-calls/latest"
API_KEY = os.environ["ROIC.AI_KEY"]


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
    return {
        "doc_id": example["id"],
        "text": example["context"],
        "metadata": {
            "task_id": example.get("task_id"),
            "source": example.get("source"),
            "difficulty": example.get("difficulty"),
            "domain": example.get("domain"),
        }
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


def main():
    source_url = os.environ["DATASET_URL"]
    raw = fetch_data(source_url)
    docs = [normalize_example(x) for x in raw]
    # load_to_db(docs)
    # build_indexes()
    # save_eval_set(docs)


if __name__ == "__main__":
    main()