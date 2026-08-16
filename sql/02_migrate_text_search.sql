BEGIN;

DROP INDEX IF EXISTS earnings_chunks_text_search_idx;

ALTER TABLE earnings_chunks
DROP COLUMN IF EXISTS text_search;

ALTER TABLE earnings_chunks
ADD COLUMN text_search tsvector
GENERATED ALWAYS AS (
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
) STORED;

CREATE INDEX earnings_chunks_text_search_idx
ON earnings_chunks
USING GIN (text_search);

COMMIT;