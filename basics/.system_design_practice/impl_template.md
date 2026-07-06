# Implementation Plan: [System / Design Name]

> The **how-to-build** companion to a `Design.md`. Where the design answers
> *what & why*, this answers *how we build it, in what order, and how we know it
> works*. Copy to `<Project>/Implementation.md` and fill the `[bracketed]` bits.
>
> **Opinionated default stack** (below) — Postgres + Redis + FastAPI — with a
> **code-owned durable-orchestration** approach (no Temporal/Kafka). It is generic
> enough for CRUD services, multi-step pipelines, and **agentic AI** systems. Two
> running worked examples are cited throughout:
> **▸ Loan** = *LoanApproval* · **▸ ACSA** = *AutonomousCustomerSupportAgent*.
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

> What are we building *in this effort*? Link the design; state the slice.

- **Implements design:** [link to Design.md + sections in scope]
- **This build delivers:** [the vertical slice / MVP being implemented now]
- **Explicitly deferred:** [design pieces out of this build]
- **Success in one sentence:** [what "done" looks like]

> **Stack vs design note.** The reference designs assume **Temporal + Kafka**.
> This stack deliberately uses **Postgres + Redis Streams** and builds durable
> orchestration in code (§8). That's a conscious trade: simpler ops, fewer
> systems, at the cost of re-implementing a slice of what Temporal gives free —
> revisit Temporal if workflows get very long-lived or very complex.

---

## 2. Prerequisites & Assumptions

| Prerequisite | Status | Owner |
|--------------|--------|-------|
| Design reviewed & approved | | |
| External accounts / credentials (bureau, KYC, LLM keys, card network) | | |
| Access to environments (dev/stage) | | |
| Upstream/downstream contracts agreed | | |

**Assumptions:** [each is a risk if wrong — cross-ref §20]

---

## 3. Tech Stack & Key Libraries

> Concrete, opinionated defaults — **nothing blank**. Swap per project, but this
> set covers CRUD, pipelines, and agentic systems.

| Layer | Choice | Notes / why |
|-------|--------|-------------|
| Language / runtime | **Python 3.12+**, asyncio | async all the way (I/O-bound: DB, Redis, LLM, external APIs) |
| API framework | **FastAPI** on **Uvicorn** (Gunicorn+uvicorn workers in prod) | async, Pydantic-native, SSE/WS support |
| Transport | **HTTPS** (TLS at nginx/ingress) · **SSE** for token/stream push · **WebSocket** for bidirectional | SSE for one-way agent streaming; WS for live chat |
| Frontend | **Jinja2** server-rendered templates (+ **HTMX** optional for partial updates / SSE) | simple, no SPA build; progressive enhancement |
| DB | **PostgreSQL 16** | ACID for money/state/audit; JSONB for flexible state; pgvector if RAG needed |
| External data | **SQL/MED — Foreign Data Wrappers** (`postgres_fdw`, `file_fdw`, others) | expose external sources (bureau/ledger read-replicas, partner DBs, files) as **foreign tables** and read/join them with plain SQL instead of bespoke clients — `IMPORT FOREIGN SCHEMA`, `CREATE FOREIGN TABLE` |
| DB access | **asyncpg** driver, **SQL-first** (SQLAlchemy Core optional for query building) | no heavy ORM; parametrized SQL; Pydantic models at the API edge |
| Migrations | **versioned SQL files** (repo convention) or Alembic (if using SQLAlchemy) | forward-only; expand/contract for zero-downtime |
| Cache | **Redis** (strings) | idempotency keys, hot config, rate-limit counters, sessions |
| Message bus / queue | **Redis Streams** (consumer groups) | durable at-least-once queue + event bus; PEL + XAUTOCLAIM for redelivery |
| Durable orchestration | **Postgres state machine + Redis Streams workers** (code-owned) | see §8 — the "no-Temporal" durable-workflow approach |
| Rate limiter | **Redis token bucket** (atomic Lua) at the API edge | bursts OK; `slowapi` as a drop-in alternative |
| Circuit breaker | **`purgatory`/`pybreaker`** or custom, **state shared in Redis** | one breaker per external dependency |
| Retries | **`tenacity`** (in-call transient) + **stream redelivery** (cross-crash) | two layers — see §8/§9 |
| Background workers | plain **async worker processes** consuming Redis Streams | scale horizontally; stateless |
| Scheduler / timers | **Redis sorted set** of due-times, or a Postgres `due_at` scan | SLA deadlines, retry-after, HIL timeouts (§8) |
| **LLM (agentic)** — pick per env | **Prod (hosted):** Anthropic **Claude Sonnet** (reasoning/tools) + **Claude Haiku** (cheap/fast). **Alt:** OpenAI GPT-class. **Local/dev:** **Ollama `qwen2.5:14b`** (reliable tool/structured output; smaller local models are not). | tiered: cheap tier for triage/drafting, large for reasoning — see §11 |
| Agent framework | **Pydantic AI** | typed tools, structured output, model-agnostic model strings, testable (`TestModel`) |
| Embeddings / RAG (if needed) | **pgvector** + an embeddings model (hosted or `nomic-embed-text` local) | keep vectors in Postgres — one datastore |
| Object storage | **S3-compatible** (presigned uploads) | documents (loan docs, statements) |
| Testing | **pytest** + **pytest-asyncio**, **testcontainers** (PG/Redis), **DeepEval** (agent eval) | §14 |
| Lint / types | **ruff** + **mypy** | CI gate |
| Observability | **structlog** (JSON), **Prometheus** metrics, **OpenTelemetry** traces, **Logfire** for Pydantic AI | §15 |
| Container / deploy | **Docker** + **docker-compose** (local), [k8s / ECS] (prod) | one image per service role (api / worker / scheduler) |

> Rule of thumb: pin exact versions in lockfiles; keep it to **two datastores
> (Postgres + Redis)** — reach for a third system only when these genuinely can't
> do the job.

---

## 4. Repository & Module Layout

> One codebase, multiple **run modes** (api / worker / scheduler) off the same image.

```
<repo>/
├── app/
│   ├── api/            # FastAPI routers (REST + SSE + WS), deps (auth, rate limit)
│   ├── web/            # Jinja2 templates + static (HTMX)
│   ├── domain/         # pure business logic + state-machine definitions (no I/O)
│   ├── orchestration/  # workflow engine: runner, steps, checkpointing, retries (§8)
│   ├── workers/        # Redis Streams consumers (one per work type)
│   ├── scheduler/      # timer/SLA scanner (§8)
│   ├── agents/         # Pydantic AI agents, tools, guardrails (§11)
│   ├── store/          # SQL models/repositories, foreign-table (SQL/MED) access, Redis clients
│   ├── integrations/   # external clients (bureau, KYC, card network, LLM) + breakers
│   ├── reliability/    # rate limiter, circuit breaker, idempotency, retry helpers (§9)
│   ├── config/         # settings (pydantic-settings), env
│   └── main.py         # entrypoints: `api`, `worker`, `scheduler`
├── migrations/         # Alembic
├── tests/              # unit / integration / e2e / eval
└── docker-compose.yml  # postgres, redis, api, worker, scheduler, nginx
```

**Layering rule:** `domain` has **no I/O imports** (pure, testable). `api`,
`workers`, `agents` orchestrate; `store`/`integrations` do I/O. Guardrails and
decision rules live in `domain` so they can't be bypassed by a model.

---

## 5. Environment & Setup

- **Local run:** `docker compose up` → nginx (TLS) → uvicorn api + N workers + scheduler + postgres + redis.
- **App entrypoints:** `python -m app.main api|worker|scheduler` (same image, different role).
- **Config & secrets:** `pydantic-settings` from env; secrets via env/secret-manager, **never in git or logs**.
- **Seed / fixtures:** `make seed` loads realistic local data + policy docs (RAG).
- **HTTPS/WS/SSE locally:** nginx terminates TLS and proxies WS upgrade + SSE (`proxy_buffering off`).

| Variable | Purpose | Default (local) |
|----------|---------|-----------------|
| `DATABASE_URL` | Postgres (asyncpg) | `postgresql+asyncpg://…` |
| `REDIS_URL` | Redis (cache + streams) | `redis://localhost:6379` |
| `LLM_PROVIDER` / `LLM_MODEL` | agent model | `ollama` / `qwen2.5:14b` |
| `LLM_API_KEY` | hosted key | — |
| `JWT_PUBLIC_KEY` | auth | — |

---

## 6. Data Layer

> The design's data model → versioned SQL migrations, **plus** the tables the
> orchestration/reliability machinery needs.

- **Schema source of truth:** versioned SQL migrations; Pydantic models at the API edge.
- **Migrations:** forward-only; **expand/contract** for zero-downtime.
- **External data via SQL/MED (Foreign Data Wrappers):** mount external sources as
  **foreign tables** and query/join them from SQL (see §7-integration note).
- **Core tables to add for this stack** (beyond the domain tables):

```sql
-- Durable workflow state (the "run") + per-step checkpoint log
CREATE TABLE workflow_runs (
    id            UUID PRIMARY KEY,
    type          TEXT NOT NULL,              -- e.g. loan_application | dispute
    status        TEXT NOT NULL,              -- running | awaiting_signal | retrying | done | failed
    current_step  TEXT NOT NULL,
    state         JSONB NOT NULL DEFAULT '{}',-- accumulated context (the checkpoint)
    attempt       INT  NOT NULL DEFAULT 0,
    next_retry_at TIMESTAMPTZ,                -- backoff schedule
    due_at        TIMESTAMPTZ,                -- SLA / HIL deadline (scheduler scans this)
    updated_at    TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON workflow_runs (status, next_retry_at);
CREATE INDEX ON workflow_runs (status, due_at);

CREATE TABLE workflow_steps (
    run_id     UUID REFERENCES workflow_runs(id),
    step       TEXT NOT NULL,
    status     TEXT NOT NULL,                 -- ok | failed
    result     JSONB,
    attempts   INT NOT NULL DEFAULT 0,
    finished_at TIMESTAMPTZ,
    PRIMARY KEY (run_id, step)
);

-- Idempotency for external side effects (durable dedup)
CREATE TABLE processed_ops (
    key        TEXT PRIMARY KEY,              -- e.g. sha256(run_id + step + op)
    result     JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Outbox: reliable event emission to Redis Streams without dual-write (see §8)
CREATE TABLE outbox (
    id           BIGSERIAL PRIMARY KEY,
    aggregate_id UUID NOT NULL,
    stream       TEXT NOT NULL,               -- target Redis stream
    payload      JSONB NOT NULL,
    published    BOOLEAN NOT NULL DEFAULT false,
    created_at   TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON outbox (published, id);

-- Immutable audit (append-only; no UPDATE/DELETE grants)
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY, actor TEXT, action TEXT, target TEXT,
    before JSONB, after JSONB, reason TEXT, ts TIMESTAMPTZ DEFAULT now()
);
```

> **▸ Loan / ▸ ACSA:** add the domain tables from each design (`loan_applications`,
> `risk_decisions` / `cases`, `disputes`, …). The four tables above are the
> stack-level machinery every multi-step design reuses.

### SQL/MED — external data as foreign tables

Use Foreign Data Wrappers to read external systems **as if they were tables**,
avoiding bespoke client code for read-heavy integrations:

```sql
CREATE EXTENSION IF NOT EXISTS postgres_fdw;
CREATE SERVER bureau_ro FOREIGN DATA WRAPPER postgres_fdw
    OPTIONS (host 'bureau-replica', dbname 'bureau', port '5432');
CREATE USER MAPPING FOR app SERVER bureau_ro OPTIONS (user 'ro', password '***');
IMPORT FOREIGN SCHEMA public LIMIT TO (credit_scores) FROM SERVER bureau_ro INTO ext;
-- now: SELECT * FROM ext.credit_scores WHERE ssn_hash = $1;   -- plain SQL join
```

**Use it for:** read federation over external replicas / warehouses / partner DBs
/ flat files — where SQL joins beat N API calls.

**Do NOT use it for** (caveats — be honest):
- **Side-effecting actions** (issue refund, credit *pull* as a billable inquiry,
  KYC). Those stay **guarded, idempotent API clients + circuit breakers** (§9) —
  an FDW read has no breaker, retry budget, or idempotency key.
- **Live OLTP externals** — point FDW at a **read replica / warehouse**, not a
  partner's production DB (load + coupling).
- Blocking assumptions: an FDW query is a **synchronous** DB call to a remote
  system — set `statement_timeout`; a slow foreign server stalls your query/worker.
- **No distributed transactions** across servers; limited predicate pushdown;
  map users least-privilege; treat foreign data as untrusted input.

---

## 7. Interface Contracts (HTTP / SSE / WS / HTML)

> FastAPI routers realize the design's API. Pull request/response bodies +
> status codes straight from the design's API section.

| Contract | Mechanism | Notes |
|----------|-----------|-------|
| REST endpoints | FastAPI routers + Pydantic schemas | `PUT` for idempotent writes; `Idempotency-Key` header |
| Streaming agent turn | **SSE** via `StreamingResponse` | `event: token/tool/done` frames |
| Live chat | **WebSocket** endpoint | for bidirectional channels |
| Server-rendered UI | **Jinja2** templates (+ HTMX for SSE-driven partials) | operator console, status pages |
| External callbacks | signed webhook routes (HMAC verify) | card network, bureau |

- **Contract-first:** FastAPI auto-generates OpenAPI; export it as the published contract.
- **Versioning:** path-prefix `/v1`; additive changes only within a version.
- **Auth:** JWT (customer) via dependency; mTLS/RBAC for admin routes; HMAC for webhooks.

---

## 8. Orchestration & Durable Execution

> **The heart of this stack.** How multi-step, long-running, crash-safe workflows
> run on **Postgres + Redis Streams** with no Temporal.

**Model: a code-owned state machine, durable in Postgres, driven by Redis Streams.**

1. **State machine (code):** each workflow `type` is a set of steps + transitions
   defined in `app/orchestration`. Steps are pure-ish functions `(state) -> result`.
2. **Durable state (Postgres):** `workflow_runs.state` is the checkpoint;
   `workflow_steps` is the per-step log.
3. **Work queue (Redis Streams):** one stream per work type; **consumer groups**
   give at-least-once delivery. `XADD` enqueue → `XREADGROUP` consume → **`XACK`**
   on success. Unacked entries sit in the **PEL**; **`XAUTOCLAIM`** reclaims
   messages from dead/stuck consumers (idle > threshold).
4. **Checkpointing:** after a step, write `workflow_steps` result **and** advance
   `current_step`/`state` in **one Postgres transaction** → that's the checkpoint.
   Crash/restart resumes from `current_step`; already-done steps are skipped.
5. **Retry loops (two layers):**
   - *In-call transient* (timeout, 503): `tenacity` exponential backoff + jitter, small max.
   - *Cross-crash / step-level*: on failure bump `attempt`, set `next_retry_at`
     (backoff); the scheduler re-enqueues when due. After `max_attempts` →
     **DLQ stream** + route to the design's fallback (human/underwriter).
6. **Long-running tasks & HIL (replaces Temporal signals/timers):** enter a
   **wait state** (`status=awaiting_signal`, `due_at=deadline`). An external event
   (approval API / webhook) updates state and `XADD`s a *resume* job. A
   **scheduler** worker scans `due_at <= now()` (or a Redis ZSET of deadlines) and
   fires SLA escalations. Waits cost nothing but a row.
7. **Multi-service orchestration:** the **orchestrator** dispatches each step as a
   job to the owning service's stream; services consume, do work, `XADD` a result
   event; the orchestrator advances the state machine. Prefer **orchestration**
   (central state machine) over choreography for auditability — matches the
   designs' "code controls flow" stance.
8. **Reliable event emission (the dual-write problem is back):** writing Postgres
   state **and** `XADD` to Redis are two systems. For events that must not be lost,
   use the **outbox**: write state + an `outbox` row in one PG transaction; a relay
   loop publishes unpublished rows to Redis Streams and marks them published
   (at-least-once → idempotent consumers). *(This is exactly the pattern that was
   unnecessary under Temporal but necessary here — see
   [`patterns/transactional_outbox.md`](patterns/transactional_outbox.md).)*

```
[API] --XADD start--> stream:orchestrator
      worker: load run (PG) -> run next step -> checkpoint (PG tx) -> XACK
                                   |  external call (idempotent, breaker)
                                   |  on retry: next_retry_at + backoff
                                   |  on wait:  status=awaiting_signal, due_at
                                   |  emit event: outbox row (same tx) -> relay -> stream
[scheduler] scans due_at/next_retry_at -> XADD resume/retry
```

> **▸ Loan:** steps `identity ‖ credit ‖ docs` (parallel) → `risk` → `route`
> (auto / underwriter-wait / deny); the underwriter queue is an `awaiting_signal`
> state with a 3-day `due_at` SLA.
> **▸ ACSA dispute:** `verify → file → provisional-credit → awaiting_signal
> (network webhook + Reg E due_at) → finalize → notify`.
>
> **Trade-off vs Temporal:** you own replay, timers, and visibility. Acceptable at
> these designs' scale; keep the state machine small and the steps idempotent.

---

## 9. Reliability Building Blocks

> Cross-cutting mechanisms every step relies on. Implement once in `app/reliability`.

| Block | Implementation | Applied where |
|-------|----------------|---------------|
| **Rate limiter** | Redis **token bucket** (atomic Lua: refill + take) as a FastAPI dependency | per tenant/API key (primary), per IP (unauth); tighter on expensive routes → `429` + `Retry-After` |
| **Circuit breaker** | per-dependency breaker, **state in Redis** (shared across workers): closed→open (N fails)→half-open (probe) | every external client (bureau, KYC, card network, LLM); on open → fallback/park |
| **Idempotency** | `Idempotency-Key` (API) in Redis `SETNX`; durable `processed_ops` for side effects | message POSTs, all external writes (refund, credit pull, ledger) |
| **Timeouts** | per-call deadline on every I/O; total per-step budget | never block a worker indefinitely |
| **Backpressure** | bounded consumer concurrency; monitor stream lag (`XLEN` - group lag); shed/queue | protect DB and downstreams under spike |
| **DLQ** | `stream:<name>:dlq` after max retries + alert | poison messages, exhausted retries |
| **Graceful shutdown** | finish in-flight, `XACK`, then exit; unacked auto-reclaimed | safe deploys |

> Rule of thumb: **every external call** is wrapped in {timeout + retry + circuit
> breaker + idempotency}. Make it a single helper so no integration forgets one.

---

## 10. Failure Points at Every Step

> Instantiate this per design (see the designs' own Fault Analysis tables). Generic
> pipeline failure modes + the stack's handling:

| Step | Failure mode | Handling |
|------|-------------|----------|
| Ingress / gateway | overload, bad auth | rate limit `429`; JWT reject `401` |
| Input validation | malformed payload | Pydantic `422`; never enqueue garbage |
| Enqueue (XADD) | Redis unavailable | retry; if state already committed, outbox relay recovers |
| Step execution | worker crash mid-step | checkpoint = last good step; resume on restart; idempotent replay |
| External call | timeout / 5xx / down | timeout + `tenacity` retry + circuit breaker → park/fallback |
| Side effect (write) | duplicate on retry | `processed_ops` idempotency key → no double action |
| Checkpoint (PG) | commit fails | step re-runs (idempotent); no partial advance |
| Event emit | Redis down after PG commit | **outbox** relay publishes later (no lost event) |
| HIL wait | approver never responds | `due_at` SLA timer → escalate/auto-action per policy |
| Retry loop | permanent failure | max attempts → DLQ + route to human |
| Completion | notify fails | notification is itself an idempotent, retried step |

---

## 11. Agentic AI Implementation

> Only if the design has LLM/agent components. Uses **Pydantic AI**; the same
> orchestration/guardrail machinery above applies.

- **Model tiers (pick per role; nothing blank):**
  - **triage / classification / drafting →** cheap tier: Claude **Haiku** (hosted) / `llama3.2:3b` or `qwen2.5:14b` (local).
  - **reasoning / tool-use / structured output →** large tier: Claude **Sonnet** (hosted) / **`qwen2.5:14b`** (local — smaller local models are unreliable at tool/structured output).
  - Model strings via Pydantic AI; tier is one config line. Default local dev = `ollama:qwen2.5:14b`.
- **Structure:** code-controlled routing (L2) → per-domain tool-calling agents
  (L3/L4). **Not** L5 model-orchestration — the orchestrator (§8) is code, for
  auditability. (▸ ACSA topology exactly.)
- **Tools:** typed Pydantic AI tools; **read** tools are free; **write** tools go
  through the deterministic **Guardrails Engine** (in `domain`) — amount/eligibility/
  legality checks the model can't override — then an idempotent side effect.
- **Structured output:** `ToolOutput(Model)` (robust) + `retries` + `temperature=0`.
- **HITL:** low-confidence / irreversible → raise an `ApprovalTask`, workflow enters
  `awaiting_signal` (§8); operator approves via admin API → resume.
- **Streaming:** stream tokens to the client over **SSE** (`StreamingResponse`).
- **Guardrails at every step + kill switch:** per-step input/output guards; a Redis
  toggle disables autonomous writes instantly (control plane).
- **Eval:** DeepEval — component + trajectory + outcome; deterministic PII/guardrail
  metrics; CI regression gate (`deepeval test run`); Logfire traces in prod.

> **▸ Loan:** LLM is **L1 LLM-as-judge** (explanation only, no routing effect) —
> shadow-deployable, zero decision-path risk. **▸ ACSA:** conversation agent (L3)
> + dispute/loan agents (L3/L4) behind guardrails + HITL.

---

## 12. Milestones (Incremental Delivery)

> Slice **vertically** — each milestone is a thin, shippable end-to-end path.

| Milestone | Delivers (vertical slice) | Exit / demo criteria |
|-----------|---------------------------|----------------------|
| M0 — Walking skeleton | one request → API → 1 workflow step → DB → response, all stubbed | request flows end-to-end |
| M1 — Happy path (1 domain) | one real workflow (▸ Loan auto-approve / ▸ ACSA balance query) | a real case resolves |
| M2 — Durability | checkpoint + retry + crash-resume + DLQ | kill a worker mid-run, it resumes |
| M3 — External + reliability | real integrations behind breaker/idempotency/rate limit | dependency outage degrades gracefully |
| M4 — HIL + timers | approval wait + SLA escalation | approve resumes; timeout escalates |
| M5 — Agentic + guardrails + eval | agents, guardrails, kill switch, eval gate | blocked action escalates; eval green |
| M6 — Observability + rollout | dashboards, alerts, CI/CD, rollback | canary + one-flag rollback works |

---

## 13. Task Breakdown & Sequencing

**Milestone [Mx]:**

| # | Task | Depends on | Parallelizable? | Est. | Owner |
|--:|------|-----------|:---------------:|------|-------|
| 1 | | — | | | |

**Critical path:** [dependency chain] · **Parallel tracks:** [independent streams]

---

## 14. Testing & Evaluation

| Layer | Covers | Tooling | Gate |
|-------|--------|---------|------|
| Unit | domain logic, guardrails, state-machine transitions | pytest | on PR |
| Integration | API↔DB↔Redis, workers, streams | pytest + testcontainers | CI |
| Contract | OpenAPI / webhook shapes | schemathesis / recorded fixtures | CI |
| Durability | crash-resume, retry, idempotency, DLQ | fault-injection integration tests | CI |
| End-to-end | full user flow (incl. SSE/WS) | httpx + async client | pre-deploy |
| Eval (AI) | intent/tool/response quality, guardrail catch, PII | **DeepEval** (`deepeval test run`) | regression gate |
| Load | NFR targets under 5× spike; stream lag | Locust / k6 | pre-release |

- **Determinism:** Pydantic AI `TestModel`/`FunctionModel` for agent unit tests;
  freeze time/uuids; seed RNG.
- **Quality gates:** ruff + mypy + coverage threshold block merge.

---

## 15. Observability & Ops Wiring

- **Logging:** `structlog` JSON; every line carries `run_id`/`case_id`, `step`, `actor`, `latency_ms`; **no PII**.
- **Metrics (Prometheus):** p95/p99 per route + per step; error rate by class; **Redis Stream lag** per group; circuit-breaker state; retry/DLQ counts; **tokens & $/case** (agentic).
- **Tracing (OTel):** one trace per request/case; spans per step, external call, agent tool. **Logfire** for Pydantic AI spans.
- **Dashboards & alerts:** stream lag, breaker-open, DLQ depth, SLA-deadline breaches, cost drift.

| Signal | Metric | Alert |
|--------|--------|-------|
| Backpressure | stream group lag | > threshold |
| Dependency down | breaker state | open |
| Stuck work | DLQ depth | > 0 rising |

---

## 16. Security & Compliance Implementation

- **AuthN/Z:** JWT (RS256) customer; mTLS + RBAC admin/control-plane; HMAC webhooks.
- **Secrets:** secret manager / env; rotation; never in code or logs.
- **Data protection:** TLS in transit; AES-256-GCM at rest for SSN/PAN; **PII masking before the LLM**.
- **Audit:** append-only `audit_log`; every state-changing action recorded (fail-closed — no action without an audit row).
- **Supply chain:** pinned deps, `pip-audit`/Dependabot, SBOM.
- **Compliance mapping:** each regulatory requirement → concrete control (▸ Loan: FCRA/ECOA adverse-action; ▸ ACSA: Reg E timers, SR 11-7 model governance).

---

## 17. Rollout & Deployment

- **CI/CD:** build → ruff/mypy → unit/integration → eval gate → image → deploy.
- **Images/roles:** one image, roles `api` / `worker` / `scheduler` (§4).
- **Migrations:** **expand/contract**, run before deploy; never break the running version.
- **Feature flags:** gate risky paths (new agent, autonomous writes); default off.
- **Release:** canary / rolling; **rollback = flag flip** (deploy ≠ release).
- **Redis Streams on deploy:** graceful shutdown `XACK`s in-flight; unacked auto-reclaimed by new consumers.

---

## 18. Operational Readiness / Runbook

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Stream lag rising | worker shortage / slow dependency | scale workers; check breaker |
| Breaker stuck open | dependency down | verify dep; manual half-open probe |
| DLQ growing | poison messages / bug | inspect, fix, replay from DLQ |
| SLA breaches | scheduler down / overload | check scheduler; scale |
| Cost spike (agentic) | prompt/loop regression | kill switch / tighten thresholds |

- **Kill switch:** Redis toggle halts autonomous writes without a deploy.

---

## 19. Performance & Cost Validation

| Target (from design NFRs) | How measured | Result |
|---------------------------|--------------|--------|
| Latency p50/p95/p99 | load test (sync path) | |
| Throughput / 5× spike | load test + stream lag | |
| Time-to-ack (async) | e2e timing | |
| Cost per case ($ / tokens) | metered run | |

---

## 20. Risks, Unknowns & Spikes

| Risk / unknown | Impact | Spike / mitigation |
|----------------|--------|--------------------|
| Home-grown orchestration edge cases | correctness | time-boxed spike: crash/retry/HIL harness in M2 before building on it |
| Redis Streams as durable bus at scale | throughput/loss | load-test lag + XAUTOCLAIM behavior early |
| Local model reliability (agentic) | flaky output | pin large tier; hosted fallback |
| Dual-write event loss | missed events | outbox relay (§8) |

---

## 21. Definition of Done / Acceptance

- [ ] In-scope milestones meet their exit criteria
- [ ] All test gates pass (unit / integration / durability / e2e / eval)
- [ ] Crash-resume, retry, idempotency, DLQ verified (fault-injection tests)
- [ ] Rate limiter, circuit breakers, timeouts on every external call
- [ ] Observability wired (logs, metrics, traces, dashboards, alerts)
- [ ] Security & compliance controls implemented and verified
- [ ] NFR targets validated (latency, throughput, cost)
- [ ] Rollback tested; runbook + kill switch documented

---

## 22. Decisions Log / Open Questions

| Date | Decision / question | Rationale / status |
|------|---------------------|--------------------|
| | | |
