import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

PG_CONN_STR = os.getenv("PG_CONN_STR")
if not PG_CONN_STR:
    raise RuntimeError("PG_CONN_STR is not set")

SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    prompt TEXT,
    model TEXT,
    response_time DOUBLE PRECISION,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost DOUBLE PRECISION,
    source TEXT DEFAULT 'earnings_rag'
);

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    feedback_type TEXT NOT NULL,
    score INTEGER,
    relevance TEXT,
    explanation TEXT
);

CREATE INDEX IF NOT EXISTS conversations_created_at_idx
    ON conversations (created_at DESC);

CREATE INDEX IF NOT EXISTS feedback_conversation_id_idx
    ON feedback (conversation_id);

CREATE INDEX IF NOT EXISTS feedback_created_at_idx
    ON feedback (created_at DESC);
"""


def main() -> None:
    with psycopg.connect(PG_CONN_STR) as conn:
        with conn.cursor() as cur:
            cur.execute(SQL)
        conn.commit()
    print("Database tables initialized successfully.")


if __name__ == "__main__":
    main()
