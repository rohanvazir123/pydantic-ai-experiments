-- Migration 008: memory system
-- Tier 2 (episodic): conversations + messages
-- Tier 3 (semantic/user): user_memories (tsvector + pgvector hybrid)
-- Tier 5 (procedural): system_prompts versioned table

-- ── Tier 2: Episodic memory ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS conversations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      TEXT NOT NULL UNIQUE,       -- crypto.randomUUID() from frontend
    tenant_id       TEXT NOT NULL,
    user_id         TEXT NOT NULL,              -- SHA-256(jwt_sub + tenant_salt)
    corpus_ids      TEXT[] NOT NULL,
    title           TEXT,                       -- first 60 chars of first user message
    summary         TEXT,                       -- auto-generated after 20 turns
    turn_count      INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_turn_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ,                -- NULL = never
    deleted_at      TIMESTAMPTZ                 -- soft delete; hard delete after 7-day grace
);

CREATE INDEX IF NOT EXISTS conversations_user_ts    ON conversations (user_id, last_turn_at DESC);
CREATE INDEX IF NOT EXISTS conversations_tenant_ts  ON conversations (tenant_id, last_turn_at DESC);
CREATE INDEX IF NOT EXISTS conversations_expires_idx ON conversations (expires_at)
    WHERE expires_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS messages (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id     UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role                TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content             TEXT NOT NULL,
    -- tsvector for full-text search within conversation history
    -- No pgvector: messages are scoped to user+conversation; BM25 alone is sufficient
    content_tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    -- Assistant-only fields (NULL on user rows)
    citations           JSONB,
    pipeline_status     TEXT,
    confidence          FLOAT,
    model_tier          TEXT,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    cost_usd            FLOAT,
    cache_hit           TEXT,
    request_id          UUID,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS messages_conversation_created ON messages (conversation_id, created_at);
CREATE INDEX IF NOT EXISTS messages_content_tsv_gin      ON messages USING GIN (content_tsv);

-- ── Tier 3: Semantic user memory ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS user_memories (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             TEXT NOT NULL,          -- SHA-256(jwt_sub + tenant_salt)
    tenant_id           TEXT NOT NULL,
    content             TEXT NOT NULL,          -- extracted fact sentence
    -- tsvector (BM25) + pgvector (cosine) — RRF hybrid search (same as entity_index)
    content_tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding           vector(768),            -- cosine ANN via HNSW
    source_message_id   UUID,                   -- message that triggered extraction
    last_retrieved_at   TIMESTAMPTZ,            -- updated on every search hit (LRU eviction)
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS user_memories_user_tenant    ON user_memories (user_id, tenant_id);
CREATE INDEX IF NOT EXISTS user_memories_tsv_gin        ON user_memories USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS user_memories_embedding_hnsw
    ON user_memories USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ── Tier 5: Procedural memory ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS system_prompts (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    content     TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    active      BOOLEAN NOT NULL DEFAULT FALSE,
    corpus_id   TEXT,           -- NULL = global; set for corpus-specific overrides
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by  TEXT NOT NULL DEFAULT 'system'
);

CREATE UNIQUE INDEX IF NOT EXISTS system_prompts_name_version ON system_prompts (name, version);
CREATE INDEX IF NOT EXISTS system_prompts_active ON system_prompts (name, active)
    WHERE active = TRUE;

-- Seed the default RAG agent system prompt (placeholder — real content in prompts.py)
INSERT INTO system_prompts (name, content, version, active, created_by)
VALUES (
    'rag_agent_v1',
    'You are a helpful assistant. Answer questions using only the provided context. '
    'Cite every factual claim with [chunk_id]. If you cannot find a supporting chunk, '
    'omit the claim entirely.',
    1,
    TRUE,
    'system'
) ON CONFLICT DO NOTHING;
