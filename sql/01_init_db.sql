CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS earnings_chunks (
    id SERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    symbol TEXT,
    year INTEGER,
    quarter INTEGER,
    date DATE,
    source TEXT,

    text_search tsvector GENERATED ALWAYS AS (
        setweight(
            to_tsvector(
                'simple',
                coalesce(symbol, '') || ' ' ||
                coalesce(year::text, '') || ' ' ||
                coalesce(quarter::text, '')
            ),
            'A'
        )
        ||
        setweight(
            to_tsvector('english', coalesce(text, '')),
            'B'
        )
    ) STORED,

    UNIQUE (doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS earnings_chunks_symbol_year_quarter_idx
ON earnings_chunks (symbol, year, quarter);

CREATE INDEX IF NOT EXISTS earnings_chunks_embedding_idx
ON earnings_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS earnings_chunks_text_search_idx
ON earnings_chunks
USING GIN (text_search);