-- Migration 006: SaaS billing — tenants, quotas, billing events

CREATE TABLE IF NOT EXISTS tenants (
    id                  TEXT PRIMARY KEY,
    display_name        TEXT NOT NULL,
    tier                TEXT NOT NULL DEFAULT 'free'
                            CHECK (tier IN ('free', 'pro', 'enterprise')),
    admin_email         TEXT NOT NULL,
    billing_customer_id TEXT,       -- Stripe customer ID
    data_region         TEXT NOT NULL DEFAULT 'us',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at          TIMESTAMPTZ -- soft delete; hard delete is async
);

CREATE TABLE IF NOT EXISTS tenant_quotas (
    tenant_id               TEXT PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
    max_queries_per_day     INTEGER NOT NULL,
    max_queries_per_minute  INTEGER NOT NULL,
    max_corpus_count        INTEGER NOT NULL,
    max_storage_gb          FLOAT NOT NULL,
    llm_enabled             BOOLEAN NOT NULL DEFAULT FALSE,
    llm_budget_usd_per_month FLOAT NOT NULL DEFAULT 0.0,
    -- 0.0 = disabled (search-only); NULL = unlimited (enterprise prepaid)
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed free-tier defaults
INSERT INTO tenants (id, display_name, tier, admin_email)
VALUES ('default', 'Default tenant', 'free', 'admin@localhost')
ON CONFLICT (id) DO NOTHING;

INSERT INTO tenant_quotas
    (tenant_id, max_queries_per_day, max_queries_per_minute, max_corpus_count, max_storage_gb, llm_enabled)
VALUES ('default', 500, 10, 1, 0.5, false)
ON CONFLICT (tenant_id) DO NOTHING;

-- Billing events (financial records — retained 7 years)
CREATE TABLE IF NOT EXISTS billing_events (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id         TEXT NOT NULL,
    corpus_id         TEXT NOT NULL,
    request_id        UUID NOT NULL,
    model_id          TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cached_tokens     INTEGER NOT NULL DEFAULT 0,
    cost_usd          FLOAT NOT NULL,
    cache_hit         TEXT,
    timestamp         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS billing_events_tenant_ts ON billing_events (tenant_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS billing_events_ts        ON billing_events (timestamp DESC);
