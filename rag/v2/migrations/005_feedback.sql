-- Migration 005: online feedback, implicit signals, token usage

CREATE TABLE IF NOT EXISTS user_feedback (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id  UUID NOT NULL,
    user_id     TEXT NOT NULL,      -- SHA-256(sub + tenant_salt)
    corpus_id   TEXT NOT NULL,
    tenant_id   TEXT NOT NULL,
    query_hash  TEXT NOT NULL,      -- SHA-256 of query text
    session_id  TEXT,
    rating      SMALLINT CHECK (rating BETWEEN 1 AND 5),
    thumbs      BOOLEAN,            -- true=up, false=down
    correction  TEXT,               -- stored as JWE on sensitive corpora
    tags        TEXT[] NOT NULL DEFAULT '{}',
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS user_feedback_corpus_submitted ON user_feedback (corpus_id, submitted_at DESC);
CREATE INDEX IF NOT EXISTS user_feedback_request_id_idx   ON user_feedback (request_id);

CREATE TABLE IF NOT EXISTS implicit_signals (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    corpus_id   TEXT NOT NULL,
    tenant_id   TEXT NOT NULL,
    signal_type TEXT NOT NULL
                    CHECK (signal_type IN (
                        'query_reformulation', 'follow_up_question',
                        'session_abandoned', 'copy_action', 'escalation'
                    )),
    request_id  UUID,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS implicit_signals_corpus_type ON implicit_signals (corpus_id, signal_type, recorded_at DESC);

-- Per-LLM-call token tracking (financial records — retained 7 years)
CREATE TABLE IF NOT EXISTS token_usage (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id        UUID NOT NULL,
    corpus_id         TEXT NOT NULL,
    tenant_id         TEXT NOT NULL,
    model_tier        TEXT NOT NULL,    -- "nano" | "small" | "large"
    model_id          TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cached_tokens     INTEGER NOT NULL DEFAULT 0,
    timestamp         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS token_usage_tenant_ts  ON token_usage (tenant_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS token_usage_corpus_ts  ON token_usage (corpus_id, timestamp DESC);
