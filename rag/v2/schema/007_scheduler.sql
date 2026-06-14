-- Migration 007: ingestion scheduler — periodic job configuration

CREATE TABLE IF NOT EXISTS scheduled_jobs (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               TEXT NOT NULL,
    name                    TEXT NOT NULL,
    source_type             TEXT NOT NULL
                                CHECK (source_type IN ('local', 'url', 's3', 'gcs')),
    source_config           JSONB NOT NULL DEFAULT '{}',
    -- e.g. {"path": "/mnt/docs"} or {"bucket": "my-bucket", "prefix": "hr/"}
    corpus_id               TEXT NOT NULL,
    cron_expr               TEXT NOT NULL,          -- standard 5-field cron (UTC)
    mode                    TEXT NOT NULL DEFAULT 'incremental'
                                CHECK (mode IN ('full', 'incremental')),
    enable_graph_extraction BOOLEAN NOT NULL DEFAULT FALSE,
    is_active               BOOLEAN NOT NULL DEFAULT TRUE,
    next_run_at             TIMESTAMPTZ,
    last_run_at             TIMESTAMPTZ,
    last_status             TEXT,                   -- "succeeded" | "failed" | "running"
    last_job_id             TEXT,                   -- Redis job hash key from last run
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS scheduled_jobs_tenant_idx    ON scheduled_jobs (tenant_id);
CREATE INDEX IF NOT EXISTS scheduled_jobs_next_run_idx  ON scheduled_jobs (next_run_at)
    WHERE is_active = TRUE;
