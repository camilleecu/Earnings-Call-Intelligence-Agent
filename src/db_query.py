import os
from dataclasses import dataclass
from typing import List, Optional

import psycopg
from dotenv import load_dotenv

load_dotenv()

PG_CONN_STR = os.getenv("PG_CONN_STR")
if not PG_CONN_STR:
    raise RuntimeError("PG_CONN_STR is not set")


@dataclass
class ConversationRecord:
    id: int
    created_at: str
    query: str
    answer: str
    prompt: Optional[str]
    model: Optional[str]
    response_time: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    cost: Optional[float]
    source: Optional[str]


@dataclass
class Stats:
    total: int
    avg_response_time: float
    total_cost: float
    avg_tokens: float
    total_feedback: int
    user_feedback_up: int
    user_feedback_down: int


def get_conversations(limit: int = 100) -> List[ConversationRecord]:
    sql = """
        SELECT
            id, created_at, query, answer, prompt, model,
            response_time, prompt_tokens, completion_tokens,
            total_tokens, cost, source
        FROM conversations
        ORDER BY created_at DESC
        LIMIT %s
    """

    with psycopg.connect(PG_CONN_STR) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()

    return [ConversationRecord(*row) for row in rows]


def get_stats() -> Stats:
    sql = """
        SELECT
            COUNT(*) AS total,
            COALESCE(AVG(response_time), 0) AS avg_response_time,
            COALESCE(SUM(cost), 0) AS total_cost,
            COALESCE(AVG(total_tokens), 0) AS avg_tokens
        FROM conversations
    """

    feedback_sql = """
        SELECT
            COUNT(*) AS total_feedback,
            COALESCE(SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END), 0) AS user_feedback_up,
            COALESCE(SUM(CASE WHEN score < 0 THEN 1 ELSE 0 END), 0) AS user_feedback_down
        FROM feedback
        WHERE feedback_type = 'user'
    """

    with psycopg.connect(PG_CONN_STR) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            total, avg_response_time, total_cost, avg_tokens = cur.fetchone()
            cur.execute(feedback_sql)
            total_feedback, user_feedback_up, user_feedback_down = cur.fetchone()

    return Stats(
        total=total,
        avg_response_time=float(avg_response_time),
        total_cost=float(total_cost),
        avg_tokens=float(avg_tokens),
        total_feedback=total_feedback,
        user_feedback_up=user_feedback_up,
        user_feedback_down=user_feedback_down,
    )
