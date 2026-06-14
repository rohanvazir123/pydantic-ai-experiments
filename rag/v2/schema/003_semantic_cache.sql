-- Migration 003: L3 semantic cache (pgvector cosine-sim lookup before LLM call)
-- Answers are stored encrypted (JWE) so tenants cannot read each other's cached responses.

CREATE TABLE IF NOT EXISTS semantic_cache (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_ids  TEXT[] NOT NULL,
    tenant_id   TEXT NOT NULL,
    query_text  TEXT NOT NULL,
    query_emb   vector(768) NOT NULL,
    answer_jwe  TEXT NOT NULL,          -- JWE-encrypted answer blob
    hit_count   INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS semantic_cache_embedding_hnsw
    ON semantic_cache USING hnsw (query_emb vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS semantic_cache_expires_idx ON semantic_cache (expires_at);
CREATE INDEX IF NOT EXISTS semantic_cache_tenant_idx  ON semantic_cache (tenant_id);
