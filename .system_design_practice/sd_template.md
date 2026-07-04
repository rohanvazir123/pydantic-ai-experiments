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

POST   /resource          — create
GET    /resource/:id      — read
PATCH  /resource/:id      — update
DELETE /resource/:id      — delete (if applicable)
```

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
