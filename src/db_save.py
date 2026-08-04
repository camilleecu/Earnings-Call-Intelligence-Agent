import os
from dataclasses import asdict
from typing import Any, Optional

import psycopg
from dotenv import load_dotenv

load_dotenv()

PG_CONN_STR = os.getenv("PG_CONN_STR")
if not PG_CONN_STR:
    raise RuntimeError("PG_CONN_STR is not set")


def save_conversation(record: Any, query: str, source: str = "earnings_rag") -> int:
    """Save one RAG interaction and return its conversation id."""
    payload = asdict(record) if hasattr(record, "__dataclass_fields__") else dict(record)

    sql = """
        INSERT INTO conversations (
            query, answer, prompt, model,
            response_time, prompt_tokens, completion_tokens,
            total_tokens, cost, source
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """

    with psycopg.connect(PG_CONN_STR) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    query,
                    payload.get("answer", ""),
                    payload.get("prompt", ""),
                    payload.get("model", "gemini-2.5-flash"),
                    payload.get("response_time"),
                    payload.get("prompt_tokens"),
                    payload.get("completion_tokens"),
                    payload.get("total_tokens"),
                    payload.get("cost"),
                    source,
                ),
            )
            conversation_id = cur.fetchone()[0]
        conn.commit()

    return conversation_id


def save_feedback(
    conversation_id: int,
    feedback_type: str,
    score: Optional[int] = None,
    relevance: Optional[str] = None,
    explanation: Optional[str] = None,
) -> None:
    """Save user or judge feedback for a conversation."""
    sql = """
        INSERT INTO feedback (
            conversation_id, feedback_type, score, relevance, explanation
        )
        VALUES (%s, %s, %s, %s, %s)
    """

    with psycopg.connect(PG_CONN_STR) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (conversation_id, feedback_type, score, relevance, explanation),
            )
        conn.commit()
