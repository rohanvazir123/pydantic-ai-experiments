-- Migration 001: initial schema
-- Documents + chunks (no corpus_id/tenant_id yet — added additively in 002)
-- pgvector and Apache AGE extensions are handled separately per container.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Full documents (metadata + raw content)
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title       TEXT NOT NULL,
    source      TEXT NOT NULL UNIQUE,
    content     TEXT,
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS documents_source_idx ON documents (source);

-- Embedded chunks (one row per chunk, FK to documents)
CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content         TEXT NOT NULL,
    embedding       vector(768),
    chunk_index     INTEGER NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}',
    token_count     INTEGER,
    content_tsv     tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_content_tsv_gin  ON chunks USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Append-only audit log (never UPDATE or DELETE)
CREATE TABLE IF NOT EXISTS audit_events (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id     TEXT NOT NULL,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    action      TEXT NOT NULL,
    corpus_id   TEXT,
    query_text  TEXT,
    request_id  UUID,
    ip_address  INET,
    response_ms INTEGER
);

CREATE INDEX IF NOT EXISTS audit_events_user_ts    ON audit_events (user_id, ts DESC);
CREATE INDEX IF NOT EXISTS audit_events_tenant_ts  ON audit_events (tenant_id, ts DESC);
