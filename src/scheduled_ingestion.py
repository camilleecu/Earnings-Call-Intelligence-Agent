"""
scheduled_ingestion.py

Kestra-friendly scheduled ingestion script for earnings call transcripts.

- Slices TRACKED_TICKERS into batches of size BATCH_SIZE.
- Picks one batch per run (based on day-of-month) to avoid manual indexing.
- Sleeps between calls to respect ROIC free-tier limits.
"""

import os
import time
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from src.data_ingestion import ingest_ticker 

load_dotenv()

# Core Nasdaq‑100 names (subset)
TRACKED_TICKERS_NASDAQ100: List[str] = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN",
    "META", "NVDA", "TSLA", "AVGO", "ADBE",
    "ADP", "INTC", "CSCO", "PEP", "COST",
]

# SOX / semiconductor names (subset)
TRACKED_TICKERS_SOX: List[str] = [
    "AMD", "ADI", "AMAT", "ASML", "AVGO", "INTC",
    "KLAC", "LRCX", "LSCC", "MRVL", "MCHP", "MU",
    "MPWR", "NVDA", "NXPI", "ON", "QRVO", "QCOM",
    "RMBS", "SWKS", "TSM", "TER", "TXN", "WOLF",
]

TRACKED_TICKERS: List[str] = sorted(set(
    TRACKED_TICKERS_NASDAQ100 + TRACKED_TICKERS_SOX
))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
SLEEP_SECONDS_BETWEEN_CALLS = int(os.getenv("SLEEP_SECONDS_BETWEEN_CALLS", "70"))


def iter_batches(seq: List[str], batch_size: int) -> List[List[str]]:
    """Split a list into consecutive batches."""
    return [seq[i:i + batch_size] for i in range(0, len(seq), batch_size)]


def choose_batch_index(num_batches: int) -> int:
    """
    Pick a batch index in a deterministic way, based on day-of-month.

    If num_batches <= 31, different days map to different batches.
    If num_batches > 31, modulo ensures a valid index but not full coverage.
    """
    today = datetime.utcnow().day  # 1–31
    return (today - 1) % num_batches if num_batches > 0 else 0


def ingest_batch(batch: List[str], batch_index: int) -> None:
    """Ingest all tickers in a single batch with rate limiting."""
    print(f"[INFO] Ingesting batch {batch_index} with tickers: {batch}")

    for j, ticker in enumerate(batch):
        try:
            print(f"[INFO] Ingesting {ticker}")
            ingest_ticker(ticker)
            print(f"[INFO] Completed {ticker}")
        except Exception as e:
            print(f"[ERROR] Failed to ingest {ticker}: {e}")
        # Rate-limiting between calls (skip sleep after last ticker)
        if j < len(batch) - 1:
            print(f"[INFO] Sleeping {SLEEP_SECONDS_BETWEEN_CALLS} seconds to respect API limits...")
            time.sleep(SLEEP_SECONDS_BETWEEN_CALLS)

    print(f"[INFO] Finished batch {batch_index}")


def main() -> None:
    """
    Entry point for Kestra / CLI.

    - Slices TRACKED_TICKERS into batches of size BATCH_SIZE.
    - Chooses one batch index based on the current day-of-month.
    - Ingests that batch with rate limiting.
    """
    batches = iter_batches(TRACKED_TICKERS, BATCH_SIZE)
    num_batches = len(batches)

    if num_batches == 0:
        print("[INFO] No tracked tickers configured; nothing to ingest.")
        return

    batch_index = choose_batch_index(num_batches)
    batch = batches[batch_index]

    print(f"[INFO] Total tracked tickers: {len(TRACKED_TICKERS)}")
    print(f"[INFO] Total batches: {num_batches}, batch_size={BATCH_SIZE}")
    print(f"[INFO] Selected batch index (day-based): {batch_index}")

    ingest_batch(batch, batch_index)


if __name__ == "__main__":
    main()



"""
 | Stage                   | What you test             | Success condition                         |
| ----------------------- | ------------------------- | ----------------------------------------- |
| Local Python            | API, Postgres, embeddings | One ticker ingests successfully           |
| Idempotency             | Repeat run                | No duplicate (doc_id, chunk_index) rows   |
| Retrieval               | Hybrid search             | New transcript chunks are returned        |
| Kestra manual execution | Container, secrets, paths | Same batch completes in Kestra            |
| Scheduled execution     | Cron trigger              | Flow launches automatically               |
| Documentation           | README/screenshots        | Pipeline is reproducible and demonstrable |
 """      