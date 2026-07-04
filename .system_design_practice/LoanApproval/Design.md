# System Design: Multi-Step Risk-Aware Loan Application Router

---

## Problem Statement

Design a multi-step, risk-aware loan application processing system for a financial
services platform operating at 1M applications/day. The system must validate
applicant identity, assess creditworthiness, verify income documentation, check
regulatory compliance by loan type and state, and route each application to the
appropriate decision lane — all with full auditability, no duplicate credit pulls on
retry, and compliance with FCRA, ECOA, and state lending regulations.

---

## Requirements

### Functional Requirements

- Accept loan applications with supporting documents (ID, pay stubs, bank statements)
- Verify applicant identity against government-issued ID
- Pull credit score and tradeline summary from a credit bureau
- Extract and validate income and employment from uploaded documents
- Compute debt-to-income (DTI) ratio
- Check regulatory rules by loan type and state
- Flag fraud indicators and sanctions/watchlist hits
- Route to one of three lanes: auto-approve, human underwriter queue, deny + compliance log
- Return a human-readable explanation with every decision
- Track application status throughout processing

### Non-Functional Requirements

| Property | Target / Notes |
|----------|---------------|
| Latency | Submission ACK < 200ms; full processing p50 < 30s, p95 < 90s, p99 < 3min |
| Availability | 99.95% — active-active across two regions |
| Consistency | Strong for decision records — financial compliance, no stale reads on outcomes |
| Scalability | 1M applications/day (~12 avg req/s; 60–120 req/s peak burst 5–10×) |
| Durability | RPO = 0 for decision records and audit log |
| Idempotency | Credit pulls, document processing must not re-execute on retry |
| Security & Compliance | GLBA, ECOA, FCRA, state lending regs; full append-only audit trail |
| Fault Tolerance | No single point of failure; partial verification results must never produce a decision |

### Out of Scope

- Loan servicing (post-approval payments, collections)
- Origination document generation (e-sign flow)
- Credit bureau integration internals

---

## Capacity Estimation

```
1M applications/day
  → ~12 req/s average
  → ~60–120 req/s peak (5–10× burst)

Each application fans out to 3 parallel verification calls:
  → ~36 req/s average to downstream services at steady state
  → ~360 req/s peak

Storage (decisions + audit log):
  ~1M rows/day × 365 = ~365M rows/year
  Each row ~2KB → ~730GB/year raw; partition by month

Queue throughput:
  ~12 jobs/s in, ~12 jobs/s out (workers must keep up at peak)
  Worker pool: size to handle 120 jobs/s peak with headroom
```

---

## Core Entities

| Entity | Key Fields | Notes |
|--------|-----------|-------|
| `LoanApplication` | id, applicant_id, loan_type, amount, state, status, created_at | Top-level aggregate |
| `LoanApplicationDetails` | application_id, ssn_encrypted, stated_income, employment_type, employer, stated_address, loan_purpose, submitted_at | Raw applicant-submitted form data — PII, encrypted at rest |
| `Applicant` | id, name, dob, ssn_hash, address | PII — encrypted at rest |
| `IdentityVerification` | application_id, status, confidence, provider_ref | Result of checking stated identity against ID provider |
| `CreditReport` | application_id, score, dti, tradelines, pulled_at | Bureau result — independent of what applicant stated |
| `IncomeVerification` | application_id, gross_income, employment_type, verified_at | Extracted from uploaded documents — compared against stated income |
| `RiskDecision` | application_id, score, tier, flags, explanation, decided_at | Output of risk synthesis agent |
| `AuditLog` | application_id, event, actor, timestamp, payload_hash | Append-only; never updated |

---

## API Design

**Interface style:** REST — decisions are resource-oriented; CRUD maps cleanly;
well-understood by all integrating partner systems.

```
POST   /applications                     — submit new application (not idempotent)
GET    /applications/:id                 — poll status
GET    /applications/:id/decision        — fetch decision + explanation
PUT    /applications/:id/documents       — upload/replace supporting docs (idempotent)
PUT    /applications/:id/withdraw        — applicant withdraws (idempotent)
```

> Prefer PUT over PATCH for update operations where retries matter.
> PATCH is not guaranteed idempotent — avoid it on operations with side effects.

**Auth:** JWT (RS256) — `tenant_id` and `applicant_id` embedded in claims.

**Idempotency header:** `Idempotency-Key` required on `POST /applications`. Replayed
requests with the same key return the original response without re-processing. Keys
stored in Redis with TTL = 24h.

---

## Infrastructure Choices

| Component | Choice | Notes |
|-----------|--------|-------|
| Database | PostgreSQL (partitioned) | ACID, pgvector, monthly range partitions |
| Message Queue | Kafka | High throughput intake; partitioned by state |
| Cache | Redis | Idempotency keys, rate limit counters, hot config |
| Workflow Engine | **Temporal** | Durable multi-step pipeline + HIL waits + SLA timers |
| Rate Limiter | Token bucket per tenant | Redis-backed; 429 + Retry-After on breach |
| Circuit Breaker | Per external dependency | One breaker each for bureau, ID provider, doc processor |

---

## High-Level Architecture

```
[Client / Partner Portal]
        │
        ▼
[API Gateway]  — JWT auth, rate limiting (token bucket, Redis), idempotency key check
        │
        ▼
[Application Service]  — writes LoanApplication (status: pending)
        │                 starts Temporal workflow execution
        ▼
[Temporal Workflow: LoanApplicationWorkflow]
        │
        ├─── [parallel activities] ───────────────────────────────────┐
        │         │                      │                            │
        ▼         ▼                      ▼                            ▼
 [Identity      [Credit Bureau      [Document Processing        (extensible:
  Activity]      Activity]           Activity — Docling]         background checks)
  L1 structured  L1 structured       L1 structured
  output         output              output
        │
        └─── [all three results assembled] ──▶ [Risk Synthesis Activity]
                                                     L3 tool-calling
                                                          │
                                                          ▼
                                                     [Router Activity]  — code
                                                     ├─ auto-approve → workflow completes
                                                     ├─ gray zone → workflow suspends
                                                     │               waits for signal
                                                     │               SLA timer running
                                                     │     ◀── underwriter_decision signal
                                                     │               workflow resumes
                                                     └─ hard fail → deny + audit log
```

---

## Data Flow & Services

1. Client submits `POST /applications`. API Gateway validates JWT, checks
   `Idempotency-Key` in Redis, applies rate limit. Returns `202 Accepted` immediately.
2. Application Service writes `LoanApplication` (status: `pending`) and **starts a
   Temporal workflow execution** (`LoanApplicationWorkflow`, keyed by `application_id`).
3. Temporal workflow **dispatches three activities in parallel** (genuinely independent):
   - **IdentityActivity** — calls ID provider; writes `IdentityVerification`. Retried
     automatically by Temporal with backoff on failure.
   - **CreditBureauActivity** — pulls score + tradelines; writes `CreditReport`.
     Idempotency key checked before bureau call — Temporal activity retries are safe
     because the key prevents a second hard inquiry.
   - **DocumentActivity** — Docling over uploaded files; writes `IncomeVerification`.
4. Workflow waits for all three activities. If any exhaust retries → workflow routes
   directly to HIL (underwriter), never auto-decides on missing data.
5. **RiskSynthesisActivity** (L3 agent) receives assembled results; runs tool loop;
   produces `RiskDecision { score, tier, flags, explanation }`.
6. **RouterActivity** (code) reads score and flags:
   - Auto-approve → workflow completes; decision written; status updated.
   - Hard fail → deny + compliance log; workflow completes.
   - **Gray zone → workflow suspends** (`workflow.wait_condition`) waiting for an
     `underwriter_decision` signal. A Temporal timer is set for the regulatory SLA
     deadline. Underwriter UI shows the pre-populated risk explanation.
7. **On signal receipt** (`PUT /applications/:id/underwriter-decision`): Application
   Service sends a Temporal signal to the running workflow. Workflow resumes, records
   the human decision, writes to `risk_decisions` and `audit_log`, completes.
8. **On SLA timer expiry** (no decision before deadline): workflow escalates
   automatically — alerts the team, optionally promotes to a senior underwriter queue.

**Sync vs async:** submission is async; workflow runs durably in Temporal workers.
Each activity is independently retried without replaying the whole pipeline. HIL waits
are free — the workflow is suspended in Temporal's durable state, consuming no threads.

---

## Agentic AI Components

| Step | Level | Tools | Who controls flow |
|------|:-----:|-------|------------------|
| Identity verification | L1 | None — single structured call | Code |
| Credit summary | L1 | None — single structured call | Code |
| Document extraction | L1 | None — single structured call | Code |
| Risk Synthesis Agent | L3 | `get_dti_ratio`, `check_regulatory_rules(state, loan_type)`, `flag_anomalies`, `lookup_fraud_indicators` | Model (bounded loop) |
| Routing decision | — | None | Code (hardcoded thresholds) |

**Autonomy boundary:** the Risk Synthesis Agent decides *which tools to call and in
what order* within a fixed bounded set. It does not control thresholds, regulatory
cutoffs, or lane assignment — those live in code. Regulators need auditable,
deterministic routing.

**Idempotency for agent side effects:** the agent's tools are read-only (lookups,
computations). The only write with a side effect (credit pull) is guarded at the
orchestrator layer before the agent runs.

**Human-in-the-loop gates:** gray-zone applications enter the underwriter queue with
the agent's explanation pre-populated. Underwriter decisions are written via
`PUT /applications/:id/underwriter-decision` (idempotent) and recorded as human
actions in the audit log. High-value loans (> $500k) always route through underwriter
regardless of score.

---

## Data Model

```sql
CREATE TABLE loan_application_details (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id      UUID NOT NULL REFERENCES loan_applications(id),
    ssn_encrypted       BYTEA NOT NULL,         -- AES-256-GCM; key in KMS
    stated_income_cents BIGINT NOT NULL,        -- gross annual, applicant-reported
    employment_type     TEXT NOT NULL,          -- 'employed' | 'self_employed' | 'unemployed' | 'retired'
    employer_name       TEXT,
    stated_address      TEXT NOT NULL,
    loan_purpose        TEXT NOT NULL,          -- 'home_purchase' | 'refinance' | 'auto' | 'personal'
    submitted_at        TIMESTAMPTZ DEFAULT now()
);
-- ssn_encrypted never leaves this table in plaintext.
-- Risk agent receives a tokenised reference, not the raw value.

CREATE TABLE loan_applications (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    applicant_id    UUID NOT NULL,
    loan_type       TEXT NOT NULL,          -- 'personal', 'mortgage', 'auto'
    amount_cents    BIGINT NOT NULL,
    state           CHAR(2) NOT NULL,
    status          TEXT NOT NULL,          -- pending | processing | decided | withdrawn
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
) PARTITION BY RANGE (created_at);         -- monthly partitions at 1M/day scale

CREATE TABLE risk_decisions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id  UUID NOT NULL REFERENCES loan_applications(id),
    score           NUMERIC(5,2) NOT NULL,
    tier            TEXT NOT NULL,          -- auto_approve | underwriter | deny
    flags           JSONB NOT NULL DEFAULT '[]',
    explanation     TEXT NOT NULL,
    decided_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE audit_log (
    id              BIGSERIAL,
    application_id  UUID NOT NULL,
    event           TEXT NOT NULL,
    actor           TEXT NOT NULL,          -- system | underwriter:<id> | applicant
    payload_hash    TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
) PARTITION BY RANGE (created_at);
-- Append-only: no UPDATE or DELETE ever issued against this table.
```

**Indexes:**
- `loan_application_details(application_id)` — join to application; one-to-one
- `loan_applications(applicant_id, created_at)` — applicant history + dedup check
- `loan_applications(status, created_at)` — queue drain and SLA monitoring queries
- `audit_log(application_id)` — full trace per application
- `risk_decisions(application_id)` — decision lookup

**Partitioning:** monthly range partitions on `created_at` for both
`loan_applications` and `audit_log`. At 1M/day, each monthly partition is ~30M rows;
old partitions can be archived to cold storage after the regulatory retention window.

---

## Test & Evaluation Framework

| Layer | What it covers | Tooling |
|-------|---------------|---------|
| Unit | DTI computation, routing thresholds, idempotency key logic | pytest |
| Integration | Orchestrator DAG, DB writes, queue publish/consume | Real DB + testcontainers |
| Contract | Credit bureau + ID provider API surface stability | Pact / recorded fixtures |
| Load | 120 req/s peak; queue drain under burst | Locust / k6 |
| Eval (AI) | Risk agent decision quality vs labeled historical outcomes; explanation coherence | LLM-as-judge + golden set |

**Regression gate for model upgrades:** run golden set before/after; block promotion
if decision tier distribution shifts > 2% or any hard-fail case is misclassified.

---

## Observability / Debuggability / Telemetry

**Metrics to alert on:**
- p95/p99 end-to-end processing latency (target < 90s / < 3min)
- Per-verification-step error rate (identity, credit, document)
- Queue depth and consumer lag (spike = worker bottleneck at scale)
- Risk agent tool call count per application (spike = looping)
- Auto-approve rate shift ± 5% week-over-week (model drift signal)
- Token spend per case

**Tracing:** one distributed trace per application — spans for queue publish/consume,
each parallel verification step, risk agent run (per-tool spans), and routing
decision.

**Logging:** structured JSON on every event:
```json
{ "application_id": "...", "step": "credit_pull", "status": "ok",
  "idempotency_key": "...", "duration_ms": 320, "ts": "..." }
```

**Runbook:** on-call checks queue lag first (worker scale-out), then per-step error
rates (dependency health), then agent tool call counts (looping). Logfire traces for
any application at p99 latency.

---

## Deep Dives

### Deep Dive 1: Idempotency for credit pulls

A hard inquiry affects the applicant's credit score and is regulated under FCRA.
Re-pulling on retry is both costly and legally problematic.

- Idempotency key = `SHA-256(application_id + "credit_pull")` written to a
  `processed_ops` table before the bureau call.
- On retry: check key → if found, return cached `CreditReport`, skip bureau call.
- Key TTL = 90 days (application processing window); after expiry, a fresh pull is
  valid (application is considered new).
- At 1M/day, the `processed_ops` table needs its own partition strategy — shard by
  `application_id % N` or use Redis with persistence for O(1) key lookup.

### Deep Dive 2: Regulatory rules by state and loan type

Rules vary: DTI limits, rate caps, required disclosures, and processing timeframes
differ by state × loan type. They change when legislatures act — not on a code
release cadence.

- **Rules engine (preferred):** versioned rules table in DB, queryable by
  `(state, loan_type, effective_date)`. Compliance team edits via an admin UI
  without a code deploy.
- The `check_regulatory_rules` tool queries this engine; the agent reasons over the
  result. The model *interprets* applicant data — it does not *store* rules.
- Rule changes are versioned; a decision always records which rule version it was
  made under (audit requirement).

### Deep Dive 3: Scaling the worker pool to 1M/day

At 120 req/s peak with ~30–90s processing time per application, the worker pool
must handle a large number of in-flight jobs concurrently.

- **Concurrency math:** if each job takes 60s (p50) and we need 120 jobs/s throughput,
  we need ~7,200 concurrent in-flight jobs at peak. Workers are I/O-bound (waiting on
  credit bureau, document processor), so each worker can handle many concurrent jobs
  via async I/O.
- **Worker design:** async Python workers (asyncio), each processing ~50–100
  concurrent jobs. At 7,200 concurrent jobs, ~72–144 worker instances.
- **Queue partitioning:** Kafka topics partitioned by `state` or `loan_type` to allow
  consumer group scaling per partition without rebalancing the whole group.
- **Backpressure:** if queue lag exceeds threshold, auto-scale worker instances
  (Kubernetes HPA on queue depth metric).

### Deep Dive 4: Temporal workflow design — HIL and long-pending tasks

**Why Temporal here:**
The loan pipeline has three properties that make a simple queue + worker fragile:
1. Multi-step with paid side effects (credit pull must not repeat on retry)
2. HIL waits that span hours to days (underwriter review)
3. Regulatory SLA deadlines that must fire automatically

Temporal solves all three with durable execution — each step is checkpointed; a
crashed worker resumes from the last completed activity, not from scratch.

**Workflow skeleton:**

```python
@workflow.defn
class LoanApplicationWorkflow:
    @workflow.run
    async def run(self, application_id: str) -> Decision:

        # Step 1: parallel verification activities
        identity, credit, docs = await asyncio.gather(
            workflow.execute_activity(identity_activity, application_id,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3)),
            workflow.execute_activity(credit_bureau_activity, application_id,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3)),
            workflow.execute_activity(document_activity, application_id,
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=3)),
        )

        # Step 2: risk synthesis
        decision = await workflow.execute_activity(
            risk_synthesis_activity, (identity, credit, docs),
            start_to_close_timeout=timedelta(seconds=120))

        # Step 3: route
        if decision.tier == "auto_approve" or decision.tier == "deny":
            return decision

        # Step 4: HIL wait — suspend workflow, set SLA timer
        self._human_decision = None
        deadline = timedelta(days=3)  # regulatory SLA

        completed = await asyncio.wait(
            [workflow.wait_condition(lambda: self._human_decision is not None),
             asyncio.sleep(deadline.total_seconds())],
            return_when=asyncio.FIRST_COMPLETED)

        if self._human_decision is None:
            # Timer fired — SLA breach, escalate
            await workflow.execute_activity(escalate_activity, application_id)
            raise ApplicationError("SLA deadline exceeded")

        return self._human_decision

    @workflow.signal
    def underwriter_decision(self, decision: Decision) -> None:
        self._human_decision = decision
```

**Key properties this gives us:**
- **Step-level retry:** credit bureau times out → Temporal retries that activity only,
  not the whole pipeline. Identity and document results are already checkpointed.
- **Free HIL wait:** workflow suspended in Temporal's persistence layer — no thread
  held, no polling. Scales to millions of pending workflows.
- **Automatic SLA enforcement:** the deadline timer is durable. Even if all workers
  restart, the timer fires and the escalation runs.
- **Full audit trail:** Temporal's event history is an append-only log of every
  activity input/output, signal received, and timer fired — queryable per workflow.
  Complements (not replaces) the application's own `audit_log` table.

**Operational note at 1M/day:** Temporal workers are stateless and horizontally
scalable. Size the worker pool to activity throughput (same math as the queue worker
pool in Deep Dive 3). Temporal Server itself needs a production deployment (clustered,
with its own PostgreSQL or Cassandra persistence) — plan for this operational overhead.

---

## Fault Analysis / Edge Cases

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Credit bureau timeout | Application stalls | Retry 3× with backoff; if all fail → underwriter queue, never auto-decide on missing data |
| Document extraction low confidence | Risk agent gets partial income data | Agent flags `low_confidence_income`; routes to underwriter |
| Risk agent hits `max_turns` | No decision produced | Route to underwriter with partial findings attached; alert on frequency |
| Duplicate submission (same applicant) | Double credit pull | Idempotency key on submission + dedup check at Application Service |
| Regulatory rules table stale | Wrong compliance check | Alert if no active rule found for (state, loan_type); block auto-approve until resolved |
| Worker crash mid-job | Job replayed from scratch | Temporal checkpoints each activity — resume from last completed step, not from scratch |
| Underwriter SLA breach | Regulatory violation | Temporal durable timer fires automatically; escalation activity runs even after worker restarts |
| Signal sent to completed workflow | Signal lost | Temporal rejects signals to closed workflows; Application Service checks workflow status before signaling |
| Temporal server down | No new workflows start; in-flight workflows pause | Temporal is clustered (HA); in-flight workflows resume automatically on recovery — durable state preserved |
| Queue consumer lag spike | Processing SLA breach | HPA auto-scales Temporal workers on activity queue depth; alert at 2× normal lag |

---

## Tradeoffs Summary

| Decision | Chosen | Alternative | Why |
|----------|--------|-------------|-----|
| Async queue-backed processing | Kafka / SQS | Sync request/response | Decouples client from 30–90s processing; handles 120 req/s burst without client timeout |
| L2 outer loop (code DAG) | Code controls flow | L5 orchestrator model | Regulators need auditable, deterministic routing; model handles interpretation only |
| Parallel verification steps | All three at once | Sequential | Genuinely independent — real latency win; critical at p95 |
| Rules engine for compliance | DB-backed versioned table | Model encodes rules | Rules change without retraining; compliance team edits directly; version tied to each decision |
| PUT over PATCH for documents | Full replace (idempotent) | Partial update | Safe to retry; simpler conflict resolution |
| Strong consistency for decisions | Synchronous DB write | Eventual | Financial decisions cannot be read stale; FCRA/ECOA audit requires point-in-time correctness |
| Monthly range partitions | Partition by created_at | Single table | 30M rows/partition manageable; old partitions archive cleanly after retention window |
| Idempotency key in Redis | Redis + TTL | DB-only | O(1) lookup at 1M/day throughput; DB `processed_ops` as durable fallback |
| Workflow engine | Temporal | Queue + worker | Multi-step durable execution, step-level retry, HIL signals, SLA timers — queue alone can't do this cleanly |
| HIL wait mechanism | Temporal signal + timer | DB polling / cron | Workflow suspends free; timer fires automatically; no polling loop to maintain |
