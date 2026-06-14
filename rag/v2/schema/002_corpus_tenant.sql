-- Migration 002: additive multi-tenant + multi-corpus columns + RLS
-- Uses ALTER TABLE ADD COLUMN … DEFAULT to avoid locking; drops defaults after backfill.

-- documents
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS corpus_id TEXT NOT NULL DEFAULT 'default',
    ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';

ALTER TABLE documents
    ALTER COLUMN corpus_id DROP DEFAULT,
    ALTER COLUMN tenant_id DROP DEFAULT;

CREATE INDEX IF NOT EXISTS documents_corpus_id_idx ON documents (corpus_id);
CREATE INDEX IF NOT EXISTS documents_tenant_id_idx ON documents (tenant_id);

-- chunks
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS corpus_id TEXT NOT NULL DEFAULT 'default',
    ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';

ALTER TABLE chunks
    ALTER COLUMN corpus_id DROP DEFAULT,
    ALTER COLUMN tenant_id DROP DEFAULT;

CREATE INDEX IF NOT EXISTS chunks_corpus_id_idx ON chunks (corpus_id);
CREATE INDEX IF NOT EXISTS chunks_tenant_id_idx ON chunks (tenant_id);

-- audit_events already has tenant_id from 001 (added as DEFAULT 'default')
ALTER TABLE audit_events ALTER COLUMN tenant_id DROP DEFAULT;

-- Row-Level Security — a connection may only see rows matching its SET LOCAL tenant_id
ALTER TABLE documents    ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks       ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_events ENABLE ROW LEVEL SECURITY;

-- Drop policies idempotently before creating (safe for re-runs)
DROP POLICY IF EXISTS tenant_isolation ON documents;
DROP POLICY IF EXISTS tenant_isolation ON chunks;
DROP POLICY IF EXISTS tenant_isolation ON audit_events;

CREATE POLICY tenant_isolation ON documents
    USING (tenant_id = current_setting('app.tenant_id', true));

CREATE POLICY tenant_isolation ON chunks
    USING (tenant_id = current_setting('app.tenant_id', true));

CREATE POLICY tenant_isolation ON audit_events
    USING (tenant_id = current_setting('app.tenant_id', true));

-- Entity shadow index table (mirrors AGE vertices for fast tsvector + pgvector search)
-- AGE does not support GIN or HNSW — this table lives in the main DB.
CREATE TABLE IF NOT EXISTS kg_entity_index (
    age_uuid        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    label           TEXT NOT NULL,
    corpus_id       TEXT NOT NULL DEFAULT 'default',
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    document_id     TEXT NOT NULL DEFAULT '',
    name_tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', name)) STORED,
    embedding       vector(768)
);

CREATE INDEX IF NOT EXISTS kg_entity_tsv_gin   ON kg_entity_index USING GIN (name_tsv);
CREATE INDEX IF NOT EXISTS kg_entity_label_idx ON kg_entity_index (label);
CREATE INDEX IF NOT EXISTS kg_entity_corpus_idx ON kg_entity_index (corpus_id, tenant_id);
CREATE INDEX IF NOT EXISTS kg_entity_embedding_hnsw
    ON kg_entity_index USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
