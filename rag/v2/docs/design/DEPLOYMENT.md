# RAG v2 — Deployment

## Table of Contents

- [Docker Compose — Local Dev](#docker-compose--local-dev)
- [Packaging & Developer Install](#packaging--developer-install)
  - [Package Manager — uv](#package-manager--uv)
  - [Optional Extras Architecture](#optional-extras-architecture)
  - [Install Script](#install-script)
  - [Latency & Safety at Install Time](#latency--safety-at-install-time)
  - [pyproject.toml Key Fields (reference)](#pyprojecttoml-key-fields-reference)
- [Cloud Deployment — Production](#cloud-deployment--production)
  - [Infrastructure Overview](#infrastructure-overview)
  - [Secrets & Config](#secrets--config)
  - [Auth in Cloud](#auth-in-cloud)
  - [Scaling Rules](#scaling-rules)
  - [Observability Stack](#observability-stack)
- [Log Storage](#log-storage)
  - [What generates logs](#what-generates-logs)
  - [Pydantic AI built-in usage tracking](#pydantic-ai-built-in-usage-tracking)
  - [Pydantic AI + Langfuse tracing](#pydantic-ai--langfuse-tracing)
  - [Where logs are stored — by environment](#where-logs-are-stored--by-environment)
  - [`backend/logs/` directory](#backendlogs-directory)
  - [Log Viewer API (UI-accessible)](#log-viewer-api-ui-accessible)
  - [CI/CD](#cicd)
- [SaaS Deployment Model](#saas-deployment-model)
  - [Tenant Isolation Model](#tenant-isolation-model)
  - [SLA Tiers](#sla-tiers)
  - [Tenant Onboarding Flow](#tenant-onboarding-flow)
  - [Quota Enforcement](#quota-enforcement)
  - [Billing & Metering](#billing--metering)
  - [Tenant Offboarding & GDPR Compliance](#tenant-offboarding--gdpr-compliance)
  - [Tenant Database Schema Additions](#tenant-database-schema-additions)

---

### Docker Compose — Local Dev

> **File:** `backend/docker-compose.yml` — this is the backend-only compose file. A top-level `docker-compose.yml` at the repo root extends it to add the `frontend` service. See TODO_implementation.md Phase 13 for the full file content.
>
> **Note on the `postgres` image:** `apache/age:latest` bundles Apache AGE but does **not** automatically include pgvector. Verify the image includes pgvector before using it, or use a custom image that installs both extensions. The existing `docker-compose.yml` at repo root maps AGE to port 5433; this design uses port 5432 — adjust if running both side by side.

```yaml
# backend/docker-compose.yml
services:
  nginx:
    image: nginx:alpine
    ports: ["443:443", "80:80"]
    volumes: [./infra/nginx/nginx.conf, ./infra/certs:/certs:ro]
    depends_on: [api]

  api:
    build: .
    command: uvicorn knowledge.api.app:app --host 0.0.0.0 --port 8000 --workers 2
    env_file: .env
    depends_on: [postgres, redis, ollama]

  ingest-worker:
    build: .
    command: python -m knowledge.ingestion.worker
    env_file: .env
    deploy:
      replicas: 2
    depends_on: [postgres, redis, ollama]

  retrieval-worker:
    build: .
    command: python -m knowledge.retrieval.worker
    env_file: .env
    deploy:
      replicas: 2
    depends_on: [postgres, redis, ollama]

  postgres:
    image: apache/age:latest              # includes pgvector + Apache AGE
    environment: [POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD]
    volumes: [pgdata:/var/lib/postgresql/data]
    ports: ["5432:5432"]

  redis:
    image: redis:7-alpine
    command: redis-server --save 60 1 --appendonly yes
    volumes: [redisdata:/data]
    ports: ["6379:6379"]

  ollama:
    image: ollama/ollama:latest
    volumes: [ollamamodels:/root/.ollama]
    ports: ["11434:11434"]
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: all, capabilities: [gpu]}]

  langfuse:        # optional observability profile
    image: langfuse/langfuse:latest
    profiles: [observability]
    depends_on: [langfuse-postgres]

  prometheus:
    image: prom/prometheus:latest
    profiles: [observability]
    volumes: [./infra/prometheus.yml:/etc/prometheus/prometheus.yml]

  grafana:
    image: grafana/grafana:latest
    profiles: [observability]
    depends_on: [prometheus]

volumes:
  pgdata:
  redisdata:
  ollamamodels:
```

**Profiles**:
- `docker compose up` — core services (api, workers, postgres, redis, ollama, nginx)
- `docker compose --profile observability up` — adds Langfuse, Prometheus, Grafana

---

### Packaging & Developer Install

The `knowledge/` module (and the current `rag/` module it replaces) is packaged as a standard Python project using **uv** and **hatchling**. The goal is a single command from a clean machine to a running system.

#### Package Manager — uv

uv replaces pip + virtualenv in one tool. Key commands:

```bash
uv sync                    # create .venv, install core deps from uv.lock
uv sync --extra all        # install every optional feature
uv sync --extra ingestion  # core + Docling ingestion only
uv run python -m rag.main  # run inside the managed venv (no activate needed)
```

The `uv.lock` file is committed to the repo. It pins every transitive dependency so any developer gets an identical environment regardless of when they clone.

#### Optional Extras Architecture

Core dependencies (always installed): Pydantic AI, asyncpg, pgvector, FastAPI, httpx. Heavy or optional features are gated behind named extras so a CI container or production image can install only what it needs.

| Extra | Key packages | When to include |
|-------|-------------|-----------------|
| `ingestion` | `docling`, `transformers` | Any node that runs `--ingest` |
| `audio` | `openai-whisper` | Audio ingestion only; also needs FFmpeg in PATH |
| `ui` | `streamlit` | Developer workstations + Streamlit deployments |
| `observability` | `langfuse` | Staging + production; not needed in CI unit tests |
| `mcp` | `mcp` | MCP server deployments only |
| `reranker` | `sentence-transformers` | API pods when `reranker_enabled = True` |
| `mem0` | `mem0ai` | API pods when `mem0_enabled = True` |
| `nl2sql` | `sqlglot` | NL-to-SQL service pods |
| `all` | everything | Local development (default) |

In Docker images, use targeted extras to keep image size down:

```dockerfile
# API image — no UI, no audio
RUN uv sync --extra ingestion --extra observability --extra reranker --extra mcp --no-dev

# Ingest-worker image
RUN uv sync --extra ingestion --extra audio --extra observability --no-dev
```

#### Install Script

Two scripts cover all platforms. Both do the same thing: install uv if missing, scaffold `.env`, run `uv sync --extra all`, and start the pgvector container.

```powershell
# Windows (PowerShell)
.\install.ps1

# Linux / macOS (Bash)
chmod +x install.sh && ./install.sh
```

After the script completes:
1. Edit `.env` — set `DATABASE_URL`, `LLM_*`, `EMBEDDING_*`
2. `ollama serve` — start Ollama
3. `ollama pull llama3.1:8b && ollama pull nomic-embed-text`
4. `uv run python -m rag.main --validate` — smoke-test the connection (v1 entrypoint; use `python -m knowledge.main --validate` once v2 is complete)
5. `uv run python -m rag.main --ingest --documents rag/documents` (v1 entrypoint; v2 uses the ingest worker via Redis)

#### Latency & Safety at Install Time

- **Guardrails and observability are off by default** (`langfuse_enabled = False`, `reranker_enabled = False`, `mem0_enabled = False`). Turn each on explicitly in `.env` once the required service is running. This prevents the install script from failing if Langfuse or a reranker endpoint isn't up yet.
- **Always measure before optimising**: run `uv run pytest rag/tests/core/ -v` first (no external deps, < 5 s). Only then run the integration suite once PostgreSQL and Ollama are confirmed healthy.
- **Specific numbers to target out of the box**: `uv sync --extra all` < 3 min on a fresh machine (dominated by Docling + Whisper downloads); `--validate` round-trip < 500 ms; first `--ingest` on the sample docs < 60 s on CPU-only Ollama.

#### pyproject.toml Key Fields (reference)

```toml
[project]
name = "rag-agent"
requires-python = ">=3.13"
# core deps here — see pyproject.toml for full list

[project.optional-dependencies]
ingestion    = ["docling>=2.14.0", "docling-core>=2.4.0", "transformers>=4.47.0"]
audio        = ["openai-whisper>=20240930"]
ui           = ["streamlit>=1.40.0"]
observability = ["langfuse>=2.0.0"]
mcp          = ["mcp>=1.0.0"]
reranker     = ["sentence-transformers>=3.0.0"]
mem0         = ["mem0ai>=0.1.0"]
nl2sql       = ["sqlglot>=25.0.0"]
all          = ["rag-agent[ingestion,audio,ui,observability,mcp,reranker,mem0,nl2sql]"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
# During migration: include both v1 (rag, kg) and v2 (knowledge) packages.
# After v2 reaches feature parity and v1 is retired: packages = ["knowledge", "nl2sql"]
packages = ["rag", "kg", "knowledge", "nl2sql"]

[tool.uv]
dev-dependencies = ["pytest>=8.3.0", "pytest-asyncio>=0.24.0", "ruff>=0.8.0", "mypy>=1.11.0"]
```

---

### Cloud Deployment — Production

#### Infrastructure Overview

```
Internet
   │
   ▼
WAF (AWS Shield / Cloudflare)
   │
   ▼
ALB / Cloud Load Balancer   ← TLS termination (ACM / GCP-managed cert)
   │
   ▼
EKS / GKE Cluster
├── Deployment: api                (2–10 pods, HPA on CPU/request rate)
├── Deployment: ingest-worker      (2–20 pods, HPA on Redis stream length)
├── Deployment: retrieval-worker   (2–10 pods, HPA on Redis stream length)
└── Istio sidecar mesh             (mTLS, traffic policies, circuit breakers)
   │
   ├── AWS Aurora PostgreSQL (Multi-AZ, pgvector enabled)
   │     └── Read replica for retrieval workers
   ├── ElastiCache Redis (Cluster Mode, 3 shards, Multi-AZ)
   ├── AGE-specific PostgreSQL (separate RDS instance or container if AGE not Aurora-compatible)
   └── S3 / GCS bucket (raw document storage, pre-signed upload URLs)
```

#### Secrets & Config

- Secrets: AWS Secrets Manager / GCP Secret Manager — DB passwords, JWT private keys, API keys.
- Config: Kubernetes ConfigMaps for non-secret settings; sealed-secrets for GitOps.
- Never pass secrets via environment variables in pod specs — use projected volumes from CSI secrets store driver.

#### Auth in Cloud

- JWT issuer: AWS Cognito User Pool (or Auth0 tenant).
- JWKS endpoint cached at API pods; key rotation handled by issuer.
- JWE keys: per-tenant RSA-OAEP keys stored in Secrets Manager; loaded at startup.
- mTLS: Istio-managed certificates (SPIFFE/SVID); zero-trust pod-to-pod.

#### Scaling Rules

| Component | Scale Trigger | Min | Max |
|---|---|---|---|
| `api` | CPU > 60% or req latency P99 > 500 ms | 2 | 10 |
| `ingest-worker` | Redis stream `knowledge:ingest` pending > 50 | 2 | 20 |
| `retrieval-worker` | Redis stream `knowledge:search` pending > 20 | 2 | 10 |
| PostgreSQL | Vertical + read replicas | — | — |
| Redis | ElastiCache shard add (manual / CloudWatch alarm) | 3 shards | 9 shards |

#### Observability Stack

- **Tracing**: OpenTelemetry SDK → AWS X-Ray / GCP Cloud Trace; Langfuse for LLM-specific traces.
- **Metrics**: Prometheus via `prometheus-client` → scrape by Grafana Cloud or CloudWatch Container Insights.
- **Logs**: structlog JSON → CloudWatch Logs / GCP Cloud Logging; correlation ID on every log line.
- **Alerts**: PagerDuty integration; alert on DLQ depth > 0, P99 search latency > 1 s, L3 cache hit rate < 20%.

See "Log Storage" section below for where logs land in local dev vs. production.

---

### Log Storage

#### What generates logs

Three distinct instrumentation layers, each stored differently:

| Layer | Tool | What it captures |
|-------|------|-----------------|
| **Structured request logs** | `structlog` (JSON) | Every HTTP request: request_id, **session_id**, user_id, tenant_id, corpus_id, route, latency_ms, status_code, cache_hit, pipeline_status, cost_usd |
| **LLM + agent traces** | Pydantic AI (built-in) + Langfuse | Every `agent.run()` / `agent.run_stream()` call: tool calls, token usage, model, latency — session_id passed as Langfuse `session_id` for conversation-level grouping |
| **Ingestion / worker logs** | `structlog` (JSON) | Job lifecycle: job_id, corpus_id, tenant_id, stage, duration_ms, chunk_count, error |
| **Alert events** | `knowledge/observability/alerts.py` | SMTP email + JSONL fallback |
| **Audit trail** | PostgreSQL `audit_events` table | Who queried what corpus when (for compliance; never deleted) |

#### Pydantic AI built-in usage tracking

Pydantic AI exposes token usage directly from every run — no manual token interception needed:

```python
# Blocking run
result = await agent.run("query", deps=state)
usage = result.usage()
# usage.request_tokens, usage.response_tokens, usage.total_tokens

# Streaming run — usage available after stream completes
async with agent.run_stream("query", deps=state) as streamed:
    async for delta in streamed.stream_text(delta=True):
        yield delta
usage = streamed.usage()
```

`RAGResponse.estimated_cost_usd` and `RAGResponse.prompt_tokens` / `completion_tokens` are populated from this usage object — not from manual token counting.

#### Pydantic AI + Langfuse tracing

We use **Langfuse** (self-hosted, open-source) for LLM traces — already in the Docker Compose setup. The `langfuse` Python SDK wraps each `agent.run()` call with a trace.

Token usage comes from Pydantic AI's built-in `result.usage()` — no manual interception, no third-party paid service:

```python
# knowledge/observability/langfuse.py
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse()   # reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from env

@observe(name="rag_agent_run")
async def traced_agent_run(query: str, ...) -> RAGResponse:
    result = await agent.run(query, deps=state)
    usage = result.usage()
    langfuse_context.update_current_observation(
        usage={"input": usage.request_tokens, "output": usage.response_tokens},
        model=settings.model_tier_small,
    )
    return build_rag_response(result, usage)
```

`RAGResponse.trace_url` is the Langfuse trace URL for that specific request (e.g. `http://localhost:3001/trace/{trace_id}`), enabling one-click jump from the UI debug panel to the full tool-call trace.

#### Where logs are stored — by environment

**Local development (Docker Compose):**

All structured logs go to **stdout**, which Docker captures per-container. Access them with:

```bash
docker compose logs -f api            # API request logs
docker compose logs -f ingest-worker  # ingestion job logs
docker compose logs -f retrieval-worker
```

Or tail all services simultaneously:

```bash
docker compose logs -f 2>&1 | grep '"level":"ERROR"'   # errors only
docker compose logs -f 2>&1 | jq -r 'select(.request_id) | [.level, .request_id, .latency_ms, .route] | @tsv'
```

**Alert fallback (local dev only):** when `SMTP_HOST` is unreachable, alerts are additionally written to `backend/logs/alerts.jsonl`. This file is in the backend file tree (`backend/logs/` directory, git-ignored). It is only a safety net — never the primary log store.

**LLM traces (local dev):** Logfire / Langfuse are optional. Run the observability Docker profile to get them locally:

```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml up langfuse
```

**Staging / Production:**

| Destination | Tool | Retention |
|-------------|------|-----------|
| Application logs (stdout) | CloudWatch Logs / GCP Cloud Logging | 30 days (configurable) |
| LLM + agent traces | Logfire cloud (or self-hosted Langfuse) | 90 days |
| Audit events | PostgreSQL `audit_events` table | 2 years |
| Token usage + billing | PostgreSQL `token_usage` + `billing_events` | 7 years |
| Metrics | Prometheus → Grafana Cloud | 13 months |

**Log format** — every structured log line is a JSON object on stdout:

```json
{
  "level": "INFO",
  "timestamp": "2026-06-07T09:23:41.123Z",
  "service": "api",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "sha256:abcd...",
  "session_id": "sess_7f3a9c2e",
  "tenant_id": "acme-corp",
  "corpus_id": "acme-corp:hr-policies",
  "route": "POST /api/v1/chat",
  "status_code": 200,
  "latency_ms": 843,
  "cache_hit": null,
  "model_tier": "small",
  "prompt_tokens": 1350,
  "completion_tokens": 287,
  "estimated_cost_usd": 0.000697,
  "pipeline_status": "answered"
}
```

**`session_id` in logs:** The frontend generates one `session_id` UUID per conversation and sends it in every `ChatRequest` body. The chat route handler extracts it and passes it to the pipeline, which injects it into the structlog context via `contextvars` before emitting the log line. It also flows into Langfuse as the `session_id` argument, grouping all traces for one conversation together. Worker and ingestion logs do not carry `session_id` (they are job-scoped, not session-scoped).

**Correlation keys:**

| Key | Scope | Where it appears |
|-----|-------|-----------------|
| `request_id` | Single HTTP request | Logs, Langfuse trace, `audit_events`, `RAGResponse.request_id`, `X-Request-ID` header |
| `session_id` | Conversation thread (N turns) | Logs, Langfuse session grouping, `conversations` table, `messages` table |
| `user_id` | All requests from one user | Logs (hashed), `audit_events`, `user_memories`, `conversations` |
| `tenant_id` | All requests from one tenant | Logs, all PostgreSQL tables (RLS), Redis quota keys |

#### `backend/logs/` directory

Added to the backend file tree for local dev use only:

```
backend/
└── logs/
    └── alerts.jsonl    # alert fallback when SMTP unreachable; git-ignored; rotated daily
```

Do not use `logs/` for application logs in production — use the container stdout pipeline.

#### Log Viewer API (UI-accessible)

The frontend provides a real-time log viewer page (`/logs`) for `admin` role users. This requires two API endpoints:

```
GET  /api/v1/logs         # query recent logs with filters (returns last N entries)
GET  /api/v1/logs/stream  # SSE stream of new log entries as they arrive (admin only)
```

**Storage:** structured log entries are written to a **Redis ring buffer** (`LPUSH knowledge:logs:recent + LTRIM 0 4999`) in addition to stdout. This gives the API fast access to the last 5,000 log lines without a DB query or file I/O. TTL is 24h per entry. The ring buffer is in Redis, not PostgreSQL — no schema migration needed.

```python
# knowledge/observability/metrics.py (addition to structlog processor chain)
class RedisLogProcessor:
    """structlog processor that mirrors each log entry to a Redis ring buffer."""
    async def __call__(self, logger, method, event_dict: dict) -> dict:
        entry = json.dumps(event_dict)
        await redis.lpush("knowledge:logs:recent", entry)
        await redis.ltrim("knowledge:logs:recent", 0, 4999)  # keep last 5000
        await redis.expire("knowledge:logs:recent", 86400)   # 24h TTL
        return event_dict
```

**Query endpoint** (`GET /v1/logs`) — on-demand, no streaming needed:

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `level` | `DEBUG\|INFO\|WARNING\|ERROR` | `INFO` | Minimum log level to return |
| `service` | string | all | Filter by `api`, `ingest-worker`, `retrieval-worker` |
| `corpus_id` | string | all | Filter by corpus |
| `request_id` | UUID | — | Return all log entries for a single request (for drilling into a specific trace) |
| `limit` | int | 100 | Max entries (capped at 500) |
| `since` | ISO timestamp | 1h ago | Only entries after this time |

Response is a JSON array of log objects, newest first. Each entry includes a `trace_url` field when the log originated from an LLM call — links directly to the Langfuse trace.

Auth: `admin` JWT role required. Logs contain hashed user IDs and corpus names — not raw PII, but not public either.

---

#### CI/CD

```
git push → GitHub Actions
  ├── ruff check + mypy + pytest (unit+mocked only)
  ├── docker build → push to ECR / Artifact Registry
  ├── helm upgrade --install (staging namespace)
  ├── smoke tests against staging
  └── manual approval gate → helm upgrade (production namespace)
```

---

### SaaS Deployment Model

The cloud deployment section describes infrastructure. This section describes the business model layered on top of it: how tenants are isolated, provisioned, billed, and offboarded. These decisions are architectural — they affect schema design, Redis key namespacing, API auth, and the K8s resource model. They must be resolved before implementation, not bolted on later.

#### Tenant Isolation Model

**Decision: Row-Level Security (RLS) on a shared PostgreSQL cluster.**

Three options considered:

| Model | Isolation | Ops overhead | Data leak risk | Decision |
|---|---|---|---|---|
| Separate cluster per tenant | Complete | Very high (N clusters) | None | Enterprise tier only |
| Schema per tenant | Strong | Medium (N schemas, DDL migrations × N) | Low (Postgres RLS supplements) | Rejected |
| Shared tables + RLS | Moderate | Low (1 schema, 1 migration) | Low if RLS is correct | **Selected for Pro/Free** |

**RLS implementation:**

```sql
-- Every data table has tenant_id TEXT NOT NULL
ALTER TABLE chunks    ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_events ENABLE ROW LEVEL SECURITY;

-- Policy: a connection may only see rows matching its set tenant_id
CREATE POLICY tenant_isolation ON chunks
    USING (tenant_id = current_setting('app.tenant_id'));

-- API sets before every query (transaction-scoped):
SET LOCAL app.tenant_id = 'acme-corp';
```

`corpus_id` format: `{tenant_id}:{corpus_slug}` — tenant is always derivable from corpus_id, giving a second isolation layer without an extra join.

**Enterprise isolation**: dedicated PostgreSQL instance + dedicated Redis namespace. Provisioned via Terraform module; not self-service.

#### SLA Tiers

| Tier | Max users | Queries/day | Rate limit | Max corpora | Storage | LLM budget/month | Price |
|---|---|---|---|---|---|---|---|
| **Free** | 5 | 500 | 10 RPM, 100 RPD | 1 | 500 MB | $0 (search-only) | $0 |
| **Pro** | 100 | 10,000 | 60 RPM, 10K RPD | 5 | 10 GB | $200 | $299/mo |
| **Enterprise** | Unlimited | Custom | Custom | Unlimited | Custom | Custom | Custom |

Free tier: LLM generation disabled. Search + cache hits only. This controls cost while still providing value.

Tier enforcement at `PRE_VALIDATE` hook:
```python
class TenantQuota(BaseModel):
    tenant_id: str
    tier: Literal["free", "pro", "enterprise"]
    max_queries_per_day: int
    max_queries_per_minute: int
    max_corpus_count: int
    max_storage_gb: float
    llm_enabled: bool                      # False for free tier
    llm_budget_usd_per_month: float        # 0.0 = unlimited (enterprise with prepaid)
    max_prompt_tokens_per_request: int = 8192
    max_output_tokens_per_request: int = 1024
```

#### Tenant Onboarding Flow

Onboarding is automated end-to-end. No manual provisioning steps.

```
1. Customer signs up (Stripe checkout)
   └── Stripe webhook → POST /v1/webhooks/stripe → subscription.created event

2. TenantProvisioner.provision(tenant_id, tier):
   a. INSERT into tenants table (id, tier, created_at, billing_customer_id)
   b. INSERT into tenant_quotas (from tier template)
   c. Generate RS256 keypair → store private key in Secrets Manager
   d. Register JWKS endpoint: GET /v1/.well-known/jwks/{tenant_id}
   e. Create default corpus: {tenant_id}:default
   f. Seed audit_events: action="tenant_provisioned"
   g. Send welcome email to admin_email (via alerts.py SMTP)

3. Customer receives:
   - API base URL: https://api.ragv2.com/api/v1
   - API key (short-lived JWT signed by tenant private key, 90-day TTL)
   - Corpus ID: {tenant_id}:default
   - Quickstart documentation link
```

Provisioning is idempotent: re-running `provision()` for an existing `tenant_id` is a no-op (all steps are `INSERT ... ON CONFLICT DO NOTHING` or check-before-execute).

#### Quota Enforcement

Quota is enforced in Redis on the hot path. PostgreSQL is the audit trail — never the enforcement gate.

```python
# knowledge/api/quota.py
async def enforce_quota(tenant_id: str, request_type: str) -> None:
    """Check and increment quota counters. Raises QuotaExceeded on breach."""
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    month = datetime.now(UTC).strftime("%Y-%m")
    minute_key = f"quota:{tenant_id}:rpm:{int(time.time() // 60)}"
    # NOTE: use datetime.now(UTC) not datetime.utcnow() — utcnow() is deprecated in Python 3.12+

    pipe = redis.pipeline()
    pipe.incr(f"quota:{tenant_id}:queries:{today}")
    pipe.expire(f"quota:{tenant_id}:queries:{today}", 86400 + 3600)  # 25h buffer
    pipe.incr(minute_key)
    pipe.expire(minute_key, 120)  # 2 min sliding window
    daily_count, _, rpm_count, _ = await pipe.execute()

    quota = await get_tenant_quota(tenant_id)  # cached in L1 for 60s

    if daily_count > quota.max_queries_per_day:
        raise QuotaExceeded(
            code="DAILY_QUOTA_EXCEEDED",
            limit=quota.max_queries_per_day,
            resets_at=next_midnight_utc(),
        )
    if rpm_count > quota.max_queries_per_minute:
        raise QuotaExceeded(
            code="RATE_LIMIT_EXCEEDED",
            limit=quota.max_queries_per_minute,
            retry_after_s=60,
        )
    if not quota.llm_enabled and request_type == "chat":
        raise QuotaExceeded(code="LLM_NOT_ENABLED_ON_FREE_TIER")
```

Quota headers on every response (even when not exceeded):
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 47
X-RateLimit-Reset: 1749214680
X-Quota-Daily-Limit: 10000
X-Quota-Daily-Used: 3241
```

#### Billing & Metering

**Billing event** emitted after every successful LLM call (async, non-blocking):

```python
class BillingEvent(BaseModel):
    id: UUID
    tenant_id: str
    corpus_id: str
    request_id: UUID
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int          # provider-level cache hits (not our L2/L3)
    cost_usd: float
    timestamp: datetime
    cache_hit: str | None       # "l2" | "l3" | None — saves tracking for cost_saved
```

Stored in `billing_events` table. Stripe usage records created nightly:

```python
# knowledge/billing/metering.py — runs as a cron job at 00:05 UTC daily
async def flush_to_stripe(date: date) -> None:
    rows = await db.fetch(
        "SELECT tenant_id, SUM(cost_usd) FROM billing_events WHERE DATE(timestamp) = $1 GROUP BY tenant_id",
        date
    )
    for tenant_id, daily_cost in rows:
        subscription_id = await get_stripe_subscription(tenant_id)
        if subscription_id:  # Pro/Enterprise tenants only
            stripe.SubscriptionItem.create_usage_record(
                subscription_item_id=subscription_id,
                quantity=int(daily_cost * 100),  # cents
                timestamp=int(datetime.utcnow().timestamp()),
            )
```

Free tier never has `subscription_id` — costs are absorbed or hard-capped at $0 (search-only). Metering events are still written for analytics.

#### Tenant Offboarding & GDPR Compliance

Data deletion is a hard requirement, not an afterthought. The system supports right-to-erasure for any tenant or individual user.

**Tenant deletion** (`DELETE /v1/tenants/{id}` — admin-only):

```python
async def delete_tenant(tenant_id: str) -> None:
    # 1. Cancel Stripe subscription immediately
    await stripe.Subscription.cancel(tenant_subscription_id)

    # 2. Cascade delete all PostgreSQL data (FK cascade handles chunks, eval_results, etc.)
    await conn.execute("DELETE FROM documents WHERE tenant_id = $1", tenant_id)
    await conn.execute("DELETE FROM gold_samples WHERE corpus_id LIKE $1", f"{tenant_id}:%")
    # semantic_cache uses corpus_ids TEXT[] (array, no FK) — must be deleted explicitly
    await conn.execute("DELETE FROM semantic_cache WHERE corpus_ids && ARRAY(SELECT id FROM corpora WHERE tenant_id = $1)", tenant_id)
    # billing_events and token_usage have no FK cascade — delete explicitly
    await conn.execute("DELETE FROM billing_events WHERE tenant_id = $1", tenant_id)
    await conn.execute("DELETE FROM token_usage WHERE tenant_id = $1", tenant_id)
    await conn.execute("DELETE FROM tenants WHERE id = $1", tenant_id)

    # 3. Delete from Apache AGE (separate connection, graph vertices/edges)
    await age_store.delete_tenant_graph(tenant_id)

    # 4. Flush Redis keys for tenant
    keys = await redis.keys(f"quota:{tenant_id}:*")
    keys += await redis.keys(f"cache:*:{tenant_id}:*")
    if keys:
        await redis.delete(*keys)

    # 5. Rotate and delete JWT private key from Secrets Manager
    await secrets_manager.delete_secret(f"jwt_private_key/{tenant_id}")

    # 6. Audit event (append-only — this row is never deleted)
    await conn.execute(
        "INSERT INTO audit_events (user_id, tenant_id, action) VALUES ($1, $2, 'tenant_deleted')",
        "system", tenant_id
    )
    # 7. Alert
    await send_alert(severity="INFO", code="TENANT_DELETED", detail={"tenant_id": tenant_id})
```

Deletion is synchronous for the PostgreSQL cascade. AGE deletion and Redis flush are background tasks with their own retry policy.

**User-level right to erasure** (`POST /v1/users/{id}/erase`):

Individual user data — `audit_events.user_id`, `user_feedback.user_id`, `implicit_signals.user_id` — is stored as `SHA-256(user_id + tenant_salt)`. Erasing a user means replacing the stored hash with `SHA-256("ERASED" + tenant_salt)`. The row structure is preserved for analytics; the user is no longer identifiable.

**Data residency**: `CorpusConfig.data_region: Literal["us", "eu", "apac"]`. Multi-region PostgreSQL routing is a Phase I IaC concern — the schema supports it from day one.

**Retention policy**: `audit_events` rows older than `AUDIT_RETENTION_DAYS` (default 2 years) are pruned by a nightly job. `user_feedback` and `implicit_signals` are retained for 1 year. `token_usage` and `billing_events` are retained for 7 years (financial records).

#### Tenant Database Schema Additions

```sql
CREATE TABLE tenants (
    id              TEXT PRIMARY KEY,           -- slug, e.g. "acme-corp"
    display_name    TEXT NOT NULL,
    tier            TEXT NOT NULL DEFAULT 'free',
    admin_email     TEXT NOT NULL,
    billing_customer_id TEXT,                  -- Stripe customer ID
    data_region     TEXT NOT NULL DEFAULT 'us',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    deleted_at      TIMESTAMPTZ                -- soft delete; hard delete is async
);

CREATE TABLE tenant_quotas (
    tenant_id               TEXT PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
    max_queries_per_day     INTEGER NOT NULL,
    max_queries_per_minute  INTEGER NOT NULL,
    max_corpus_count        INTEGER NOT NULL,
    max_storage_gb          FLOAT NOT NULL,
    llm_enabled             BOOLEAN NOT NULL DEFAULT false,
    llm_budget_usd_per_month FLOAT NOT NULL DEFAULT 0.0,
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE billing_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL,
    corpus_id       TEXT NOT NULL,
    request_id      UUID NOT NULL,
    model_id        TEXT NOT NULL,
    prompt_tokens   INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cached_tokens   INTEGER NOT NULL DEFAULT 0,
    cost_usd        FLOAT NOT NULL,
    cache_hit       TEXT,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ON billing_events (tenant_id, timestamp DESC);
CREATE INDEX ON billing_events (timestamp DESC);   -- for daily flush job

-- Per-LLM-call token tracking (source of truth for cost; retained 7 years)
CREATE TABLE token_usage (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id        UUID NOT NULL,
    corpus_id         TEXT NOT NULL,
    tenant_id         TEXT NOT NULL,
    model_tier        TEXT NOT NULL,     -- "nano" | "small" | "large"
    model_id          TEXT NOT NULL,     -- exact model name
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cached_tokens     INTEGER NOT NULL DEFAULT 0,  -- provider-level prompt cache hits
    timestamp         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ON token_usage (tenant_id, timestamp DESC);
CREATE INDEX ON token_usage (corpus_id, timestamp DESC);
```

---

