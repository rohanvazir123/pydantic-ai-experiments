# Implementation Plan: [System / Design Name]

> The **how-to-build** companion to a `Design.md`: *how we build it, in what order,
> and how we know it works*. Copy to `<Project>/Implementation.md` and fill the
> `[bracketed]` bits. Default stack: **Postgres + Redis + FastAPI** with
> **code-owned durable orchestration** (no Temporal/Kafka) — fits CRUD, pipelines,
> and agentic AI. Worked examples: **▸ Loan** (*LoanApproval*) · **▸ ACSA**
> (*AutonomousCustomerSupportAgent*).
>
> **Implements:** [`Design.md`](./Design.md) · **Owner:** [name] · **Status:** draft

## Table of Contents

- [1. Overview & Scope](#1-overview--scope)
- [2. Prerequisites & Assumptions](#2-prerequisites--assumptions)
- [3. Tech Stack & Key Libraries](#3-tech-stack--key-libraries)
- [4. Repository & Module Layout](#4-repository--module-layout)
- [5. Environment & Setup](#5-environment--setup)
- [6. Data Layer](#6-data-layer)
- [7. Interface Contracts (HTTP / SSE / WS / HTML)](#7-interface-contracts-http--sse--ws--html)
- [8. Orchestration & Durable Execution](#8-orchestration--durable-execution)
- [9. Reliability Building Blocks](#9-reliability-building-blocks)
- [10. Failure Points at Every Step](#10-failure-points-at-every-step)
- [11. Agentic AI Implementation](#11-agentic-ai-implementation)
- [12. Milestones (Incremental Delivery)](#12-milestones-incremental-delivery)
- [13. Task Breakdown & Sequencing](#13-task-breakdown--sequencing)
- [14. Testing & Evaluation](#14-testing--evaluation)
- [15. Observability & Ops Wiring](#15-observability--ops-wiring)
- [16. Security & Compliance Implementation](#16-security--compliance-implementation)
- [17. Rollout & Deployment](#17-rollout--deployment)
- [18. Operational Readiness / Runbook](#18-operational-readiness--runbook)
- [19. Performance & Cost Validation](#19-performance--cost-validation)
- [20. Risks, Unknowns & Spikes](#20-risks-unknowns--spikes)
- [21. Definition of Done / Acceptance](#21-definition-of-done--acceptance)
- [22. Decisions Log / Open Questions](#22-decisions-log--open-questions)

---

## 1. Overview & Scope

- **Implements design:** [link to Design.md + sections in scope]
- **This build delivers:** [the vertical slice / MVP being implemented now]
- **Explicitly deferred:** [design pieces out of this build]
- **Success in one sentence:** [what "done" looks like]

> **Stack note.** Reference designs assume **Temporal + Kafka**; this stack uses
> **Postgres + Redis Streams** and builds durable orchestration in code (§8) —
> simpler ops, fewer systems, at the cost of re-implementing a slice of Temporal.
> Revisit Temporal if workflows get very long-lived or complex.

---

## 2. Prerequisites & Assumptions

| Prerequisite | Status | Owner |
|--------------|--------|-------|
| Design reviewed & approved | | |
| External credentials (bureau, KYC, LLM keys, card network) | | |
| Access to dev/stage environments | | |
| Upstream/downstream contracts agreed | | |

**Assumptions:** [each is a risk if wrong — cross-ref §20]

---

## 3. Tech Stack & Key Libraries

Opinionated defaults — swap per project.

| Layer | Choice |
|-------|--------|
| Language / runtime | **Python 3.12+**, asyncio |
| API | **FastAPI** on **Uvicorn** (Gunicorn+uvicorn workers in prod) |
| Transport | **HTTPS** (TLS at ingress) · **SSE** for streaming · **WebSocket** for bidirectional |
| Frontend | **Jinja2** templates (+ **HTMX** optional) |
| DB | **PostgreSQL 16** (ACID; JSONB; pgvector if RAG) |
| External data | **SQLModel over an external read-replica engine** + guarded API clients (§9) — replicas via a *second* async engine; side-effecting integrations via typed API clients, not direct DB access |
| DB access | **SQLModel** (Pydantic-native ORM over SQLAlchemy) on **asyncpg** — one class = table + API-edge schema; drop to SQLAlchemy Core / raw SQL for hot paths |
| Migrations | **Alembic** (autogenerate from SQLModel metadata); forward-only, expand/contract |
| Cache | **Redis** (idempotency keys, hot config, rate-limit counters, sessions) |
| Queue / bus | **Redis Streams** (consumer groups; PEL + XAUTOCLAIM for redelivery) |
| Durable orchestration | **Postgres state machine + Redis Streams workers** (code-owned, §8) |
| Rate limiter | **Redis token bucket** (atomic Lua); `slowapi` alt |
| Circuit breaker | `purgatory`/`pybreaker` or custom, **state in Redis**, one per dependency |
| Retries | `tenacity` (in-call) + stream redelivery (cross-crash) — §9 |
| Workers | plain **async processes** consuming Redis Streams; scale horizontally |
| Scheduler / timers | Redis ZSET of due-times or Postgres `due_at` scan (SLA/HIL) |
| **LLM (agentic)** | **Prod:** Claude **Sonnet** (reasoning) + **Haiku** (cheap). **Local:** Ollama **`qwen2.5:14b`** (smaller local models unreliable at tools). |
| Agent framework | **Pydantic AI** (typed tools, structured output, `TestModel`) |
| Embeddings / RAG | **pgvector** + embeddings model — keep vectors in Postgres |
| Object storage | **S3-compatible** (presigned uploads) |
| Testing | **pytest** + **pytest-asyncio**, **testcontainers**, **DeepEval** |
| Lint / types | **ruff** + **mypy** |
| Observability | **structlog** (JSON), **Prometheus**, **OpenTelemetry**, **Logfire** for Pydantic AI |
| Deploy | **Docker** + **compose** (local), [k8s/ECS] (prod); one image per role |

> Pin versions in lockfiles. Keep to **two datastores (Postgres + Redis)** — reach
> for a third only when these genuinely can't do the job.

---

## 4. Repository & Module Layout

One codebase, multiple **run modes** (api / worker / scheduler) off the same image.

```
<repo>/
├── app/
│   ├── api/            # FastAPI routers (REST + SSE + WS), deps (auth, rate limit)
│   ├── web/            # Jinja2 templates + static (HTMX)
│   ├── domain/         # pure business logic + state-machine defs (NO I/O)
│   ├── orchestration/  # workflow engine: runner, steps, checkpointing, retries (§8)
│   ├── workers/        # Redis Streams consumers
│   ├── scheduler/      # timer/SLA scanner (§8)
│   ├── agents/         # Pydantic AI agents, tools, guardrails (§11)
│   ├── store/          # SQLModel models + repositories, external read-replica engine, Redis clients
│   ├── integrations/   # external clients + breakers
│   ├── reliability/    # rate limiter, breaker, idempotency, retry (§9)
│   ├── config/         # pydantic-settings
│   └── main.py         # entrypoints: api | worker | scheduler
├── migrations/
├── tests/
└── docker-compose.yml
```

**Layering rule:** `domain` has **no I/O imports** (pure, testable). `api`/`workers`/
`agents` orchestrate; `store`/`integrations` do I/O. Guardrails + decision rules live
in `domain` so a model can't bypass them.

---

## 5. Environment & Setup

- **Local run:** `docker compose up` → nginx (TLS) → uvicorn api + N workers + scheduler + postgres + redis.
- **Entrypoints:** `python -m app.main api|worker|scheduler` (same image, different role).
- **Config/secrets:** `pydantic-settings` from env; secrets via secret-manager, **never in git or logs**.
- **Seed:** `make seed` loads realistic data + policy docs (RAG).

| Variable | Purpose | Default (local) |
|----------|---------|-----------------|
| `DATABASE_URL` | Postgres (asyncpg) | `postgresql+asyncpg://…` |
| `REDIS_URL` | Redis | `redis://localhost:6379` |
| `LLM_PROVIDER` / `LLM_MODEL` | agent model | `ollama` / `qwen2.5:14b` |
| `JWT_PUBLIC_KEY` | auth | — |

---

## 6. Data Layer

- **Source of truth:** **SQLModel** classes (Pydantic-native ORM) — same class is the table *and* the API-edge schema. **Alembic** autogenerates migrations from their metadata; forward-only, expand/contract (review every revision).
- **External data:** map read-only external replicas (bureau/ledger/partner DBs) to SQLModel models on a **second async engine**; side-effecting externals stay guarded API clients (§9).
- **Stack machinery tables** (beyond domain tables): durable workflow state +
  per-step checkpoint, idempotency dedup, outbox, append-only audit.

```sql
-- Durable run + checkpoint log
CREATE TABLE workflow_runs (
    id UUID PRIMARY KEY, type TEXT NOT NULL,
    status TEXT NOT NULL,          -- running | awaiting_signal | retrying | done | failed
    current_step TEXT NOT NULL,
    state JSONB NOT NULL DEFAULT '{}',   -- the checkpoint
    attempt INT NOT NULL DEFAULT 0,
    next_retry_at TIMESTAMPTZ, due_at TIMESTAMPTZ,   -- backoff / SLA (scheduler scans)
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON workflow_runs (status, next_retry_at);
CREATE INDEX ON workflow_runs (status, due_at);

CREATE TABLE workflow_steps (
    run_id UUID REFERENCES workflow_runs(id), step TEXT NOT NULL,
    status TEXT NOT NULL, result JSONB, attempts INT NOT NULL DEFAULT 0,
    finished_at TIMESTAMPTZ, PRIMARY KEY (run_id, step)
);

CREATE TABLE processed_ops (          -- durable idempotency for side effects
    key TEXT PRIMARY KEY, result JSONB, created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE outbox (                 -- reliable event emission (§8)
    id BIGSERIAL PRIMARY KEY, aggregate_id UUID NOT NULL, stream TEXT NOT NULL,
    payload JSONB NOT NULL, published BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON outbox (published, id);

CREATE TABLE audit_log (              -- append-only; no UPDATE/DELETE grants
    id BIGSERIAL PRIMARY KEY, actor TEXT, action TEXT, target TEXT,
    before JSONB, after JSONB, reason TEXT, ts TIMESTAMPTZ DEFAULT now()
);
```

> **▸ Loan / ▸ ACSA:** add each design's domain tables; the four above are shared machinery.

### SQLModel — models, repositories, and external reads

One `SQLModel` class is both the ORM table and the Pydantic schema:

```python
class LoanApplication(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    status: str = Field(index=True)
    amount: Decimal
    state: dict = Field(default_factory=dict, sa_column=Column(JSONB))
    created_at: datetime = Field(default_factory=datetime.utcnow)

async def get_run(session: AsyncSession, run_id: uuid.UUID) -> LoanApplication | None:
    return await session.get(LoanApplication, run_id)
```

Read-only external replicas → their own SQLModel models on a **second async engine**
(`create_async_engine(BUREAU_REPLICA_URL)`), read with the same ORM.

**Boundary (be honest):**
- Reads (domain + external replicas) → SQLModel repositories.
- Hot-path / analytical → drop to SQLAlchemy Core / raw SQL.
- **Side-effecting externals** (refund, billable credit *pull*, KYC) → guarded idempotent API clients + breakers (§9), never an ORM write into someone else's DB.
- No cross-database joins (two engines) — join in app code or cache locally first.
- Set `statement_timeout` on the external engine; least-privilege/read-only; treat external data as untrusted.

---

## 7. Interface Contracts (HTTP / SSE / WS / HTML)

| Contract | Mechanism | Notes |
|----------|-----------|-------|
| REST | FastAPI routers + Pydantic schemas | `PUT` for idempotent writes; `Idempotency-Key` header |
| Streaming turn | **SSE** via `StreamingResponse` | `event: token/tool/done` |
| Live chat | **WebSocket** | bidirectional |
| Server UI | **Jinja2** (+ HTMX for SSE partials) | operator console, status pages |
| External callbacks | signed webhooks (HMAC verify) | card network, bureau |

- **Contract-first:** FastAPI auto-generates OpenAPI → publish it.
- **Versioning:** path-prefix `/v1`; additive-only within a version.
- **Auth:** JWT (customer); mTLS/RBAC admin; HMAC webhooks.

---

## 8. Orchestration & Durable Execution

**The heart of this stack:** a code-owned state machine, durable in Postgres, driven
by Redis Streams — multi-step, long-running, crash-safe, no Temporal.

1. **State machine (code):** each workflow `type` = steps + transitions; steps are pure-ish `(state) -> result`.
2. **Durable state (PG):** `workflow_runs.state` = checkpoint; `workflow_steps` = per-step log.
3. **Queue (Redis Streams):** one stream per work type; consumer groups → at-least-once. `XADD` → `XREADGROUP` → **`XACK`**; **`XAUTOCLAIM`** reclaims dead consumers' PEL entries.
4. **Checkpointing:** after a step, write `workflow_steps` result **and** advance `current_step`/`state` in **one PG transaction**. Crash resumes from `current_step`; done steps skipped.
5. **Retries (two layers):** in-call transient → `tenacity` backoff+jitter; cross-crash → bump `attempt`, set `next_retry_at`, scheduler re-enqueues. After `max_attempts` → **DLQ** + fallback (human).
6. **Long-running / HIL:** enter `status=awaiting_signal`, `due_at=deadline`. External event (approval/webhook) updates state + `XADD`s a resume job; a **scheduler** scans `due_at <= now()` and fires SLA escalations. Waits cost only a row.
7. **Multi-service:** the orchestrator dispatches each step to the owning service's stream; services do work + `XADD` a result event; orchestrator advances. Prefer orchestration over choreography (auditability).
8. **Reliable event emission (dual-write):** PG state + `XADD` are two systems. Write state + an `outbox` row in one PG tx; a relay publishes unpublished rows → Redis (at-least-once → idempotent consumers). See [`patterns/transactional_outbox.md`](patterns/transactional_outbox.md).

```
[API] --XADD start--> stream:orchestrator
  worker: load run (PG) -> run step -> checkpoint (PG tx) -> XACK
             | external call (idempotent, breaker)
             | retry: next_retry_at + backoff   | wait: status=awaiting_signal, due_at
             | emit: outbox row (same tx) -> relay -> stream
[scheduler] scans due_at/next_retry_at -> XADD resume/retry
```

> **▸ Loan:** `identity ‖ credit ‖ docs` → `risk` → `route` (auto / underwriter-wait / deny);
> underwriter queue = `awaiting_signal` + 3-day `due_at`.
> **▸ ACSA dispute:** `verify → file → provisional-credit → awaiting_signal (webhook + Reg E due_at) → finalize → notify`.
> **Trade-off vs Temporal:** you own replay, timers, visibility — keep the state machine small and steps idempotent.

---

## 9. Reliability Building Blocks

Implement once in `app/reliability`.

| Block | Implementation | Applied where |
|-------|----------------|---------------|
| Rate limiter | Redis token bucket (Lua) as FastAPI dep | per tenant/key (+ per IP); `429` + `Retry-After` |
| Circuit breaker | per-dependency, state in Redis (closed→open→half-open) | every external client; on open → fallback/park |
| Idempotency | `Idempotency-Key` in Redis `SETNX`; durable `processed_ops` | POSTs, all external writes |
| Timeouts | per-call deadline + per-step budget | never block a worker forever |
| Backpressure | bounded concurrency; monitor stream lag; shed/queue | protect DB + downstreams |
| DLQ | `stream:<name>:dlq` after max retries + alert | poison / exhausted messages |
| Graceful shutdown | finish in-flight, `XACK`, exit; unacked auto-reclaimed | safe deploys |

> **Every external call** = {timeout + retry + circuit breaker + idempotency}. Make it one helper.

---

## 10. Failure Points at Every Step

| Step | Failure | Handling |
|------|---------|----------|
| Ingress | overload / bad auth | rate limit `429`; JWT `401` |
| Validation | malformed payload | Pydantic `422`; never enqueue garbage |
| Enqueue | Redis down | retry; outbox relay recovers if state committed |
| Step exec | worker crash mid-step | checkpoint = last good step; idempotent replay |
| External call | timeout / 5xx | timeout + retry + breaker → park/fallback |
| Side effect | duplicate on retry | `processed_ops` key → no double action |
| Checkpoint | commit fails | step re-runs (idempotent); no partial advance |
| Event emit | Redis down post-commit | outbox relay publishes later |
| HIL wait | approver never responds | `due_at` SLA → escalate/auto-action |
| Retry loop | permanent failure | max attempts → DLQ + human |
| Completion | notify fails | notification is itself an idempotent retried step |

---

## 11. Agentic AI Implementation

Only if the design has LLM/agent components. Uses **Pydantic AI**; same orchestration/
guardrail machinery applies.

- **Model tiers:** triage/draft → cheap (Haiku / small local); reasoning/tools/structured → large (Sonnet / `qwen2.5:14b`). Tier = one config line. Default local dev = `ollama:qwen2.5:14b`.
- **Structure:** code-controlled routing (L2) → per-domain tool-calling agents (L3/L4). **Not** L5 model-orchestration — the orchestrator (§8) is code, for auditability.
- **Tools:** typed Pydantic AI tools; **read** free; **write** goes through the deterministic **Guardrails Engine** (`domain`) → idempotent side effect.
- **Structured output:** `ToolOutput(Model)` + `retries` + `temperature=0`.
- **HITL:** low-confidence / irreversible → `ApprovalTask`, workflow `awaiting_signal` (§8) → operator approves → resume.
- **Streaming:** tokens over SSE.
- **Guardrails + kill switch:** per-step input/output guards; a Redis toggle disables autonomous writes instantly.
- **Eval:** DeepEval (component + trajectory + outcome; deterministic PII/guardrail metrics; CI gate `deepeval test run`); Logfire traces in prod.

> **▸ Loan:** LLM = L1 narrator (explanation only, no routing effect) — shadow-deployable.
> **▸ ACSA:** conversation agent (L3) + dispute/loan agents (L3/L4) behind guardrails + HITL.

---

## 12. Milestones (Incremental Delivery)

Slice **vertically** — each milestone is a thin, shippable end-to-end path.

| Milestone | Delivers | Exit criteria |
|-----------|----------|---------------|
| M0 — Walking skeleton | request → API → 1 stubbed step → DB → response | flows end-to-end |
| M1 — Happy path | one real workflow (▸ Loan auto-approve / ▸ ACSA balance) | a real case resolves |
| M2 — Durability | checkpoint + retry + crash-resume + DLQ | kill a worker mid-run, it resumes |
| M3 — External + reliability | real integrations behind breaker/idempotency/rate limit | outage degrades gracefully |
| M4 — HIL + timers | approval wait + SLA escalation | approve resumes; timeout escalates |
| M5 — Agentic + guardrails + eval | agents, guardrails, kill switch, eval gate | blocked action escalates; eval green |
| M6 — Observability + rollout | dashboards, alerts, CI/CD, rollback | canary + one-flag rollback |

---

## 13. Task Breakdown & Sequencing

**Milestone [Mx]:**

| # | Task | Depends on | Parallel? | Est. | Owner |
|--:|------|-----------|:---------:|------|-------|
| 1 | | — | | | |

**Critical path:** [chain] · **Parallel tracks:** [independent streams]

---

## 14. Testing & Evaluation

| Layer | Covers | Tooling | Gate |
|-------|--------|---------|------|
| Unit | domain logic, guardrails, transitions | pytest | on PR |
| Integration | API↔DB↔Redis, workers, streams | pytest + testcontainers | CI |
| Contract | OpenAPI / webhook shapes | schemathesis / fixtures | CI |
| Durability | crash-resume, retry, idempotency, DLQ | fault-injection | CI |
| E2E | full user flow (incl. SSE/WS) | httpx async client | pre-deploy |
| Eval (AI) | intent/tool/response, guardrail catch, PII | DeepEval | regression gate |
| Load | NFR under 5× spike; stream lag | Locust / k6 | pre-release |

- **Determinism:** Pydantic AI `TestModel`/`FunctionModel`; freeze time/uuids; seed RNG.
- **Quality gates:** ruff + mypy + coverage block merge.

---

## 15. Observability & Ops Wiring

- **Logging:** `structlog` JSON; every line carries `run_id`/`case_id`, `step`, `actor`, `latency_ms`; **no PII**.
- **Metrics (Prometheus):** p95/p99 per route + step; error rate by class; **Redis Stream lag**; breaker state; retry/DLQ counts; **tokens & $/case**.
- **Tracing (OTel):** one trace per request/case; spans per step/call/tool. **Logfire** for Pydantic AI.

| Signal | Metric | Alert |
|--------|--------|-------|
| Backpressure | stream group lag | > threshold |
| Dependency down | breaker state | open |
| Stuck work | DLQ depth | > 0 rising |

---

## 16. Security & Compliance Implementation

- **AuthN/Z:** JWT (RS256) customer; mTLS + RBAC admin; HMAC webhooks.
- **Secrets:** secret manager / env; rotation; never in code or logs.
- **Data:** TLS in transit; AES-256-GCM at rest for SSN/PAN; **PII masking before the LLM**.
- **Audit:** append-only `audit_log`; fail-closed (no action without an audit row).
- **Supply chain:** pinned deps, `pip-audit`/Dependabot, SBOM.
- **Compliance mapping:** each requirement → a concrete control (▸ Loan: FCRA/ECOA adverse-action; ▸ ACSA: Reg E timers, SR 11-7 governance).

---

## 17. Rollout & Deployment

- **CI/CD:** build → ruff/mypy → unit/integration → eval gate → image → deploy.
- **Roles:** one image, roles `api` / `worker` / `scheduler`.
- **Migrations:** expand/contract, run before deploy; never break the running version.
- **Flags:** gate risky paths (new agent, autonomous writes); default off. **Rollback = flag flip.**
- **Redis Streams on deploy:** graceful shutdown `XACK`s in-flight; unacked auto-reclaimed.

---

## 18. Operational Readiness / Runbook

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Stream lag rising | worker shortage / slow dep | scale workers; check breaker |
| Breaker stuck open | dependency down | verify dep; manual half-open probe |
| DLQ growing | poison / bug | inspect, fix, replay from DLQ |
| SLA breaches | scheduler down / overload | check scheduler; scale |
| Cost spike (agentic) | prompt/loop regression | kill switch / tighten thresholds |

- **Kill switch:** Redis toggle halts autonomous writes without a deploy.

---

## 19. Performance & Cost Validation

| Target (from NFRs) | How measured | Result |
|--------------------|--------------|--------|
| Latency p50/p95/p99 | load test (sync path) | |
| Throughput / 5× spike | load test + stream lag | |
| Time-to-ack (async) | e2e timing | |
| Cost per case ($ / tokens) | metered run | |

---

## 20. Risks, Unknowns & Spikes

| Risk / unknown | Impact | Spike / mitigation |
|----------------|--------|--------------------|
| Home-grown orchestration edge cases | correctness | time-boxed crash/retry/HIL harness in M2 first |
| Redis Streams as durable bus at scale | throughput/loss | load-test lag + XAUTOCLAIM early |
| Local model reliability (agentic) | flaky output | pin large tier; hosted fallback |
| Dual-write event loss | missed events | outbox relay (§8) |

---

## 21. Definition of Done / Acceptance

- [ ] In-scope milestones meet exit criteria
- [ ] All test gates pass (unit / integration / durability / e2e / eval)
- [ ] Crash-resume, retry, idempotency, DLQ verified (fault-injection)
- [ ] Rate limiter, breakers, timeouts on every external call
- [ ] Observability wired (logs, metrics, traces, dashboards, alerts)
- [ ] Security & compliance controls implemented and verified
- [ ] NFR targets validated (latency, throughput, cost)
- [ ] Rollback tested; runbook + kill switch documented

---

## 22. Decisions Log / Open Questions

| Date | Decision / question | Rationale / status |
|------|---------------------|--------------------|
| | | |
