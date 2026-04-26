-- =============================================================================
-- docker/init-db.sql
-- =============================================================================
-- Executed automatically by PostgreSQL on first container start.
-- Loads the AGE and pgvector extensions, then bootstraps the graph and the
-- movie_embeddings table required by CineGraph-AI.
-- =============================================================================

-- 1. Load extensions
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS vector;

-- Make AGE functions available in the current search path
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- 2. Create the Apache AGE graph (idempotent via DO block)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'movies_graph'
    ) THEN
        PERFORM ag_catalog.create_graph('movies_graph');
        RAISE NOTICE 'Graph "movies_graph" created.';
    ELSE
        RAISE NOTICE 'Graph "movies_graph" already exists. Skipping.';
    END IF;
END;
$$;

-- 3. Create the pgvector embeddings table
--    const   : IMDb ID (tt...) — links graph node to vector row
--    plot    : source text used to generate the embedding
--    embedding: 1536-dim vector from OpenAI text-embedding-3-small
--    metadata: JSONB bag for title, year, rating (display fields)
CREATE TABLE IF NOT EXISTS public.movie_embeddings (
    id        SERIAL PRIMARY KEY,
    const     TEXT        NOT NULL UNIQUE,  -- IMDb ID, e.g. tt0268978
    plot      TEXT        NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    metadata  JSONB        DEFAULT '{}'::JSONB
);

-- 4. HNSW index for fast approximate nearest-neighbour search
--    Parameters are conservative for a ~800-row dataset.
--    Scale m / ef_construction upward if dataset grows beyond 100k rows.
CREATE INDEX IF NOT EXISTS movie_embeddings_hnsw_idx
    ON public.movie_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 5. Stored procedure: semantic search with optional IMDb ID pre-filter
--    Follows pgvector.mdc: "Keep database logic separate from application logic"
CREATE OR REPLACE FUNCTION public.search_movie_embeddings(
    query_embedding VECTOR(1536),
    candidate_ids   TEXT[]   DEFAULT NULL,  -- NULL = no graph pre-filter
    top_k           INT      DEFAULT 10
)
RETURNS TABLE (
    const       TEXT,
    plot        TEXT,
    metadata    JSONB,
    distance    FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        me.const,
        me.plot,
        me.metadata,
        (me.embedding <=> query_embedding)::FLOAT AS distance
    FROM movie_embeddings me
    WHERE
        candidate_ids IS NULL
        OR me.const = ANY(candidate_ids)
    ORDER BY me.embedding <=> query_embedding
    LIMIT top_k;
END;
$$ LANGUAGE plpgsql;
