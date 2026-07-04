# System Design: [Problem Title]

---

## Problem Statement

> One paragraph. What system are we building, for whom, and what problem does it solve?

---

## Requirements

### Functional Requirements

- 
- 
- 

### Non-Functional Requirements

| Property | Target / Notes |
|----------|---------------|
| Latency | p50 / p95 / p99 targets |
| Availability | SLA (e.g. 99.9%) — note CAP/PACELC tradeoff |
| Consistency | Strong / eventual / causal — justify the choice |
| Scalability | Expected load; how it grows |
| Durability | Data loss tolerance (RPO) |
| Idempotency | Which operations must be safe to retry |
| Security & Compliance | Auth model, data sensitivity, regulatory constraints |
| Fault Tolerance | Failure modes the system must survive |

### Out of Scope

- 

---

## Core Entities

> The nouns of the system. Keep it tight — only entities that own state.

| Entity | Key Fields | Notes |
|--------|-----------|-------|
| | | |

---

## API Design

> Pick the right style for the use case; don't default to REST blindly.

**Interface style:** REST / GraphQL / gRPC / event-driven — justify

```
# Key endpoints / messages / mutations

POST   /resource          — create (not idempotent — each call creates a new resource)
GET    /resource/:id      — read
PUT    /resource/:id      — full replace (idempotent — safe to retry)
DELETE /resource/:id      — delete (idempotent — deleting twice is the same as once)
```

> Prefer PUT over PATCH for update operations where retries matter.
> PATCH is not guaranteed idempotent — avoid it on operations with side effects.

**Auth:** JWT / API key / mTLS

---

## High-Level Architecture

> Diagram in ASCII or prose. Services, their responsibilities, and how they connect.

```
[Client]
   │
   ▼
[API Gateway / Auth]
   │
   ├─▶ [Service A]  — responsibility
   ├─▶ [Service B]  — responsibility
   └─▶ [Service C]  — responsibility
         │
         ▼
      [Data Store]
```

---

## Data Flow & Services

> Walk through the main request path end-to-end. Be specific about sync vs async.

1. 
2. 
3. 

**Sync vs async decision:** justify which operations block and which are queued.

---

## Agentic AI Components

> Only if the system includes LLM/agent workflows. Skip if not applicable.

| Agent / Step | Level (L1–L5) | Tools | Who controls flow |
|-------------|:---:|-------|------------------|
| | | | |

**Autonomy boundary:** what the model decides vs what code enforces.

**Idempotency for agent side effects:** how retries are made safe.

**Human-in-the-loop gates:** which decisions require approval and at what threshold.

---

## Data Model

> Schema sketches for the key entities. Enough to reason about queries and indexes.

```sql
-- Example
CREATE TABLE resource (
    id          UUID PRIMARY KEY,
    tenant_id   UUID NOT NULL,
    status      TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);
```

**Indexes:** what queries need to be fast?

**Partitioning / sharding strategy:** if relevant at scale.

---

## Infrastructure Choices

| Component | Choice | Notes |
|-----------|--------|-------|
| Database | | |
| Message Queue | | |
| Cache | | |
| Workflow Engine | | |
| Rate Limiter | | |
| Circuit Breaker | | |

---

### Database

| Option | When to choose |
|--------|---------------|
| PostgreSQL | Default — relational, ACID, pgvector for embeddings, good at joins and transactions |
| DynamoDB / Cassandra | Write-heavy, wide column, high throughput key-value at scale |
| Redis (primary) | Ephemeral state only — not a source of truth |

**Choice for this system:**

**Partitioning strategy:**

### Message Queue

| Option | When to choose |
|--------|---------------|
| Kafka | High throughput, durable, replay, fan-out to multiple consumers |
| SQS / SQS FIFO | Simpler ops; FIFO for ordering guarantees; AWS-native |
| RabbitMQ | Lower throughput, complex routing rules |

**Choice for this system:**

**Partition / topic design:**

**Backpressure handling:** how does the system slow down producers when consumers lag?

### Redis Caching

| Use case | TTL | Eviction policy |
|----------|-----|----------------|
| Idempotency keys | 24h | none (explicit delete on expiry) |
| Session / auth tokens | per JWT exp | allkeys-lru |
| Hot read cache (e.g. rules, config) | 5–60min | allkeys-lru |
| Rate limit counters | per window | volatile-ttl |

**What is NOT cached:** source-of-truth financial records, audit log, decision records.

### Workflow Engine (if multi-step processing required)

Use when: the processing pipeline has multiple sequential steps, each of which can
fail independently, and you need durable retry, step-level checkpointing, and
visibility into in-flight workflows without building it yourself.

| Option | When to choose |
|--------|---------------|
| **Temporal** | Long-running workflows, durable execution, step-level replay, strong consistency guarantees. Best for workflows that span minutes to days and touch paid side effects. |
| **Prefect / Airflow** | Data pipelines, scheduled batch jobs, DAG-oriented work |
| **Celery** | Shorter async task queues, Python-native, simpler ops. Good for job scheduling and background tasks that don't need durable multi-step state. |

**Choice for this system:**

**Why:** justify based on workflow duration, failure recovery needs, and operational
complexity tolerance.

> Rule of thumb: if a failed step needs to resume from where it left off (not restart
> from scratch), and the steps touch external paid side effects, reach for Temporal.
> If it's fire-and-forget background tasks or scheduled jobs, Celery is sufficient.

### Rate Limiter

Prevents any single client or tenant from overwhelming the system. Apply at the API
Gateway layer before requests hit any service.

| Algorithm | When to choose |
|-----------|---------------|
| **Token bucket** | Allows short bursts above the rate; good for APIs where clients occasionally spike |
| **Fixed window** | Simplest; suffers boundary burst (2× rate at window edge) |
| **Sliding window log** | Precise; memory-heavy at scale |
| **Sliding window counter** | Good balance — approximate but cheap |

**Scope of limiting:**
- Per tenant / API key (primary)
- Per IP (secondary, for unauthenticated endpoints)
- Per endpoint (e.g. tighter limits on expensive operations like document upload)

**Storage:** Redis counters with TTL = window size. Atomic `INCR` + `EXPIRE`.

**On limit exceeded:** return `429 Too Many Requests` with `Retry-After` header.
Never silently drop — the client must know to back off.

**Choice for this system:**

### Circuit Breaker

Prevents cascading failure when an external dependency (credit bureau, ID provider,
downstream service) is degraded or down. Without it, slow dependencies cause thread
exhaustion and latency amplification across the whole system.

**States:**

```
CLOSED ──(N consecutive failures)──▶ OPEN ──(cooldown elapsed)──▶ HALF-OPEN
  ▲                                                                     │
  └──────────────────(probe succeeds)──────────────────────────────────┘
                      (probe fails → back to OPEN)
```

| State | Behavior |
|-------|---------|
| **Closed** | Normal operation; failures are counted |
| **Open** | Fail fast immediately; no calls to the dependency |
| **Half-open** | Allow one probe request; success → Closed, failure → Open |

**Per dependency:** one breaker per external integration (credit bureau, ID provider,
document processor, LLM endpoint). Do not share a breaker across unrelated services.

**Failure threshold:** tune per dependency — a credit bureau that times out 3×
in 30s is down; an LLM endpoint may need a higher threshold for transient errors.

**On open circuit:** route to the fallback path (underwriter queue, cached response,
degraded mode) — never surface a raw error to the client if a safe fallback exists.

**Choice for this system:**

---

## Test & Evaluation Framework

| Layer | What it covers | Tooling |
|-------|---------------|---------|
| Unit | Business logic, routing, schemas | pytest / Jest |
| Integration | Service boundaries, DB, queues | Real deps or testcontainers |
| Contract | API surface stability | Pact / OpenAPI |
| Load | Throughput, latency under pressure | Locust / k6 |
| Eval (AI) | Model output quality, confidence calibration | LLM-as-judge, golden set |

**Regression gate for model upgrades:** how you detect decision boundary shifts.

---

## Observability / Debuggability / Telemetry

**Metrics to alert on:**
- p95/p99 latency per service and per agent step
- Error rate by type (validation, external dependency, model)
- Token spend per case (for agentic flows)
- Queue depth / consumer lag

**Tracing:** distributed trace per request — what spans are emitted?

**Logging:** structured JSON; what fields on every log line?

**Runbook hooks:** what does an on-call engineer look at first?

---

## Deep Dives

> Pick 2–3 hard sub-problems to go deep on. Each gets its own section.

### Deep Dive 1: [Topic]

### Deep Dive 2: [Topic]

### Deep Dive 3: [Topic]

---

## Fault Analysis / Edge Cases

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Dependency X goes down | | Circuit breaker + fallback |
| Retry causes duplicate action | | Idempotency key |
| Model returns malformed output | | Validation + retry budget |
| Context window exceeded | | Trim + checkpoint |
| | | |

---

## Tradeoffs Summary

> One table. The decisions you made and what you gave up.

| Decision | Chosen | Alternative | Why |
|----------|--------|-------------|-----|
| | | | |
