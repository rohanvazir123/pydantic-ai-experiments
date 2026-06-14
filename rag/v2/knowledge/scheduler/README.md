# knowledge/scheduler/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [How It Works](#how-it-works)
- [Source Adapters](#source-adapters)

---

## What This Is

Periodic ingestion scheduler. Allows corpora to be kept up to date automatically — a scheduled job scans a source (local folder, URL, S3, GCS) and publishes ingest jobs to Redis every time new or changed files are found.

---

## Files

| File | Purpose |
|------|---------|
| `runner.py` | `APScheduler` integration: polls `get_due_jobs()` every 60s; publishes `IngestJob` to Redis stream |
| `job_store.py` | `ScheduledJob` CRUD against `scheduled_jobs` table; `compute_next_run_at()` via `croniter` |
| `schemas.py` | `ScheduledJob`, `JobTrigger`, `JobStatus` Pydantic models |

---

## How It Works

1. Admin creates a scheduled job via `POST /v1/scheduler/jobs` with a cron expression and source config.
2. `runner.py` (APScheduler tick, every 60s) calls `job_store.get_due_jobs()` — jobs where `next_run_at <= NOW() AND is_active = TRUE`.
3. For each due job, an `IngestJob` is published to the `knowledge:ingest` Redis stream.
4. `job_store.update_next_run_at()` advances the cron to the next fire time.
5. The ingest worker picks up the job from the stream (same path as manual ingest).

**Incremental mode**: the worker checks the SHA-256 fingerprint cache before processing each file. Unchanged files are skipped — only new or modified files are re-ingested.

---

## Source Adapters

| Adapter | Config | Dependency |
|---------|--------|-----------|
| `LocalFolderSource` | `{"path": "/mnt/docs"}` | None |
| `URLSource` | `{"url": "https://..."}` | None |
| `S3Source` | `{"bucket": "my-bucket", "prefix": "hr/"}` | `boto3` (optional extra) |
| `GCSSource` | `{"bucket": "my-bucket", "prefix": "hr/"}` | `google-cloud-storage` (optional extra) |
