# System Design: Autonomous Customer Support Agent for Financial Inquiries

> Derived from [`design_notes.md`](design_notes.md); structured with
> [`../sd_template.md`](../sd_template.md). Architecture decisions here are made
> for *this* system, not copied from `LoanApproval`.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Requirements](#requirements)
- [Core Entities](#core-entities)
- [API Design](#api-design)
- [High-Level Architecture](#high-level-architecture)
- [Data Flow & Services](#data-flow--services)
- [Agentic AI Components](#agentic-ai-components)
- [Guardrails at Every Step](#guardrails-at-every-step)
- [Data Model](#data-model)
- [Infrastructure Choices](#infrastructure-choices)
- [Test & Evaluation Framework](#test--evaluation-framework)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Observability](#observability--debuggability--telemetry)
- [Deep Dives](#deep-dives)
- [Fault Analysis / Edge Cases](#fault-analysis--edge-cases)
- [Tradeoffs Summary](#tradeoffs-summary)

---

## Problem Statement

> A conversational support agent for a bank that **resolves complex, multi-step
> financial inquiries end-to-end** — disputed charges and loan applications being
> the flagship cases — with **no human in the common path**. It answers simple
> account questions synchronously and drives long-running cases (disputes,
> loans) to resolution asynchronously. Autonomy is **bounded and delegated**: the
> agent acts freely within a safe, read-mostly envelope; irreversible or
> low-confidence actions go through deterministic guardrails and human approval.
> Humans retain override control at all times via an operator control plane.

**Why "bounded" and not "full" autonomy:** in banking, full autonomy over
irreversible actions is a liability, constrained by regulation — ECOA /
fair-lending + adverse-action notices (loans), Reg E / Reg Z error-resolution
rules (disputes), SR 11-7 model-risk governance — plus fraud and security risk.
The design is deliberately partial autonomy behind strict guardrails and human
fail-safes.

---

## Requirements

### Functional Requirements

- **Converse** multi-turn across channels (web/chat, mobile, email, IVR) and
  maintain per-conversation context.
- **Answer simple inquiries synchronously** — balance, transaction status/history,
  policy/T&C questions (RAG), card status — using read-only tools.
- **Resolve disputed charges** end-to-end: intake → validate transaction →
  file/track network dispute → provisional credit within the Reg E window →
  final resolution → notify.
- **Process loan applications** end-to-end: intake + document collection → KYC /
  identity → credit-bureau pull → eligibility/risk assessment → decision
  (auto-approve within limits / refer to underwriter / decline with adverse-action
  notice).
- **Escalate to humans** with fully populated context for irreversible or
  low-confidence actions; resume automatically on approval/rejection.
- **Enforce deterministic guardrails** on every state-changing action (limits,
  eligibility, legality) — override the model when it proposes an invalid action.
- **Operator control plane** — kill switch, runtime risk thresholds, review /
  override queue, per-tool/per-tenant toggles.
- **Immutable audit trail** of every decision and action (agent, human, system).

### Non-Functional Requirements

| Property | Target / Notes |
|----------|---------------|
| Latency | **Simple (sync):** p95 < 6 s, first-token < 1.5 s (streamed). **Complex (async):** time-to-ack < 3 s; time-to-resolution is case-bound (disputes up to the Reg E window; loans minutes→days). |
| Throughput | 50k tickets/day (~0.6/s avg, ~2–3/s peak); absorb **5× spike → 250k/day** (~3/s avg, ~10–15/s peak). |
| Availability | 99.9% for the conversational + intake plane. Case workflows are **durable** — a worker outage delays but never loses a case. |
| Consistency | **Strong** for money movement, case state, and the audit log (Postgres, ACID). **Eventual** for conversation working memory and read caches. |
| Durability | RPO ≈ 0 for cases/audit/decisions (no data loss). Conversation working memory is best-effort (reconstructable from the message log). |
| Idempotency | Every external write (network dispute, ledger credit, bureau pull, notification) must be safe to retry via an orchestrator-issued idempotency key. |
| Security & Compliance | PII masked/tokenized **before** the LLM and scrubbed from logs; encryption at rest (AES-256-GCM for SSN/PAN); ECOA/Reg E/Reg Z/SR 11-7; immutable audit. |
| Fault Tolerance | Survive: LLM/tool/dependency outages, worker crashes mid-workflow, malformed model output, reasoning-loop stalls, duplicate deliveries. |
| Cost | Avg **$0.08 / ticket**, held by routing the bulk to a cheap fast-track and reserving large models for complex reasoning. |

### Out of Scope

- Human agent desktop UX beyond the approval/override queue API.
- Fraud *detection* modeling (we consume a fraud/risk signal; we don't build it).
- Core-banking ledger, card-network, and credit-bureau systems themselves (external, integrated via APIs).
- Marketing/upsell and non-support conversations.

---

## Core Entities

> Only entities that own state. Two planes: **conversation** (fast, ephemeral)
> and **case** (durable, source of truth).

| Entity | Key Fields | Notes |
|--------|-----------|-------|
| `Conversation` | id, customer_id, channel, status, created_at | The live dialogue; working memory in Redis. |
| `Message` | id, conversation_id, role, content (PII-masked), tokens, ts | Append-only turn log; lets working memory be rebuilt. |
| `Case` | id, conversation_id, customer_id, type (`dispute`\|`loan`\|…), status, risk_tier, workflow_id, created_at | Durable unit of work; owns a Temporal workflow. |
| `Dispute` | case_id, txn_id, amount, reason_code, network_ref, provisional_credit, reg_e_deadline, status | Dispute-specific state + regulatory timer. |
| `LoanApplication` | case_id, product, amount, applicant_details_ref, bureau_ref, decision, adverse_action_ref | Loan-specific; stated data separate from verified. |
| `CustomerAccount` | customer_id, accounts[], kyc_status | **System-of-record (external)** — read via API, never copied into the vector store. |
| `AgentStep` | id, scope (conv/case), n, thought, tool_call, tool_result, tokens, latency | Per-step trace for audit + telemetry. |
| `ApprovalTask` | id, case_id, kind, payload, status, operator_id, decided_at | HITL work item; blocks a durable wait. |
| `IdempotencyKey` | key, case_id, tool, request_hash, response, status | Dedup store for external writes. |
| `AuditRecord` | id, actor, action, target, before, after, reason, ts | **Immutable, append-only**; regulatory record. |
| `PolicyDoc` | id, version, text, embedding | Semantic memory (RAG) for policies/T&Cs/rules. |

---

## API Design

**Interface style:** **REST + streaming (SSE)** for the customer-facing
conversation (turn-based, needs token streaming); **event-driven internally**
(case events on a bus); **signed webhooks** for external callbacks (card network,
bureau). gRPC between internal services optional; not customer-facing.

```
# Customer plane (JWT auth, rate-limited, PII-masked at the edge)
POST /v1/conversations                      → { conversation_id }         (create)
POST /v1/conversations/{id}/messages        → SSE stream of assistant turn (Idempotency-Key)
                                              may emit { case_id } when a case opens
GET  /v1/conversations/{id}                 → conversation + latest state
GET  /v1/cases/{id}                         → case status / timeline       (customer-visible subset)

# Internal (service-to-service; PUT = idempotent, safe to retry)
PUT  /internal/cases/{id}                   → upsert case state
PUT  /internal/cases/{id}/steps/{n}         → record an agent step

# External callbacks (HMAC-signed, verified at the edge)
POST /v1/webhooks/card-network              → dispute lifecycle update
POST /v1/webhooks/bureau                    → async credit report ready

# Operator control plane (separate service, mTLS + RBAC, fully audited)
POST /admin/kill-switch                     → halt all autonomous actions (scope: global|tool|tenant)
PUT  /admin/policy/thresholds               → runtime risk limits / confidence cutoffs
GET  /admin/review-queue                    → escalations + sampled auto-resolutions
POST /admin/approvals/{task_id}             → approve | reject (resumes the workflow)
```

> **PUT over PATCH** for all internal state updates so retries are safe.
> **Idempotency-Key** required on message POSTs (a retried customer turn must not
> spawn a second case).
> The **operator control plane is a physically separate service** with its own
> authz — the agent can never call it.

**Auth:** customer → JWT (short-lived); internal → mTLS; webhooks → HMAC
signature; operators → mTLS + RBAC.

---

## High-Level Architecture

Two planes. The **conversational plane** is real-time and stateless-per-request
(state in Redis). The **case plane** is durable and long-running (Temporal). The
**control plane** sits above both so humans keep control.

```
Channels (web / mobile / email / IVR)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ API Gateway  — authN, rate limit (token bucket), PII MASKING   │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────┐     working memory
│ Conversation Service       │◀────────────────▶ [ Redis ]
│  observe→think→act loop     │
│  (stateless workers)        │──RAG──▶ [ pgvector: policies ]
└───────────────────────────┘
        │  intent + risk tier (Triage: cheap model)
        ├────────────── simple / read-only ──────────────┐
        │                                                 ▼
        │                                   Conversation Agent (L3)
        │                                   read-only tools → stream reply
        │
        └── complex (dispute / loan) ─▶ open Case
                                          │
                                          ▼
                         ┌───────────────────────────────────┐
                         │ Case Orchestrator  = Temporal       │
                         │ durable workflow: steps, retries,   │
                         │ SLA timers, HITL waits, checkpoints  │
                         └───────────────────────────────────┘
                             │ activities (each guarded + idempotent)
        ┌────────────────────┼───────────────────────┬───────────────┐
        ▼                    ▼                        ▼               ▼
  Dispute Agent (L3/L4)  Loan Agent (L3/L4)   Policy/RAG        Notifier
        │                    │                        │               │
        ▼  every write ▶ [ Policy & Guardrails Engine ] ▶ [ Idempotency store ]
        ▼
  External deps via CIRCUIT BREAKERS:
  card network · core-banking ledger · credit bureau · KYC/ID · doc processor · LLM

        ▲                                            │
        │ approve / reject / pause / override        ▼
┌──────────────────────────────────────────────────────────────┐
│ CONTROL PLANE  — kill switch · thresholds · review queue        │  (mTLS+RBAC)
└──────────────────────────────────────────────────────────────┘
        │
        ▼
[ PostgreSQL: cases, decisions, IMMUTABLE AUDIT LOG ]   [ Kafka: case/audit events ]
```

---

## Data Flow & Services

**A. Simple query (synchronous — the fast track).** *e.g. "What's my balance?"*

1. Message → gateway (authN, rate limit, **PII masking**) → Conversation Service.
2. Load working memory (Redis); **Triage** (cheap model) classifies intent + risk tier.
3. Intent is read-only → **Conversation Agent** runs a bounded tool loop
   (account lookup, transaction status, or policy RAG) and **streams** the answer.
4. Append masked turn to the message log; update working memory. **No case opened.**
   Target p95 < 6 s.

**B. Complex case (asynchronous — durable).** *e.g. "I didn't make this $2,400 charge"*

1–2. Same intake + triage; intent = `dispute` (or `loan`), risk tier ≥ 2.
3. Conversation Agent collects the minimum to open the case, then **starts a
   Temporal workflow** and returns an **ack + case_id** (time-to-ack < 3 s). The
   conversation is now free; updates arrive asynchronously.
4. The **Case Orchestrator** runs the domain workflow as durable activities:
   - **Dispute:** verify transaction → check eligibility (Reg E) → file with card
     network → **issue provisional credit within the regulatory window (SLA
     timer)** → await network outcome (days, via webhook) → finalize → notify.
   - **Loan:** collect docs → KYC/identity → **credit-bureau pull (idempotent)** →
     eligibility/risk assessment → decision (auto-approve in limits / refer to
     underwriter / decline + adverse-action notice).
5. Every state-changing activity passes the **Guardrails Engine** and uses an
   **idempotency key**. Irreversible/low-confidence steps raise an
   **ApprovalTask** and the workflow **durably waits** (Temporal signal).
6. On completion: notify the customer (push/email/next turn), update CRM, write
   the **audit record**, emit a case event to Kafka.

**Sync vs async decision:** anything answerable read-only from live data returns
inline. Anything that (a) touches money, (b) calls a slow external system
(network/bureau), or (c) needs HITL is a **durable async case** — never blocked on
in the request path.

---

## Agentic AI Components

| Agent / Step | Level | Tools | Who controls flow |
|-------------|:---:|-------|------------------|
| Triage / intent + risk classifier | **L1** | none (single classify) | Code dispatches on the label |
| Top-level routing | **L2** | — | **Code** (deterministic, auditable) |
| Conversation Agent (simple Q&A) | **L3** | read-only: balance, txn status, card status, policy RAG | Model sequences tools within a turn |
| Dispute Agent | **L3/L4** | txn lookup, dispute-file, provisional-credit, network status, RAG | Model within the workflow step; **workflow (code) owns the lifecycle** |
| Loan Agent | **L3/L4** | doc intake, KYC, bureau pull, eligibility calc, RAG | Same — model interprets, code + rules decide |
| Drafting (notices, replies) | **L1/L2** | templates + RAG | Code-templated, model fills |

**This is a code-orchestrated multi-agent system (L2 routing over L3/L4 domain
agents) — deliberately *not* L5 model-orchestration.** A model never decides
cross-domain delegation; the router and the Temporal workflow (both code) do.
Rationale: determinism and auditability are regulatory requirements — see the
[deep dive](#deep-dive-1-two-plane-architecture--why-not-l5-model-orchestration).

**Autonomy boundary — what the model decides vs. what code enforces:**

| Model may decide | Code / rules enforce (model cannot override) |
|------------------|----------------------------------------------|
| Which read-only tool to call, and phrasing | Whether a *write* is allowed (limits, eligibility, legality) |
| How to interpret a policy for a case | The final loan/dispute **decision rules** (regulated) |
| What to ask the customer next | Routing between domains; when to escalate |
| A *proposed* refund/credit amount | The cap on that amount + idempotency + audit |

**Idempotency for agent side effects:** the orchestrator issues a key per logical
action (`case_id + action_type + attempt_of`); the idempotency store makes a
retried network file / ledger credit / bureau pull a no-op that returns the prior
result.

**Human-in-the-loop gates:** trigger on (a) irreversible/financial-impact actions
above the runtime threshold, or (b) intent/decision **confidence < 75%**. The
workflow serializes state, raises an `ApprovalTask`, **durably waits**, then
resumes with the human decision injected into context.

**Deterministic fallbacks:** if the model emits a malformed/unschema'd action, an
illegal action, or stalls (loop detection), code takes over — retry within budget,
then escalate. The model is never the last line of defense on money movement.

---

## Guardrails at Every Step

Guardrails are **not** a single engine in front of writes — every step in the
pipeline has an **input guard**, an **output guard**, and a **defined violation
action**. The principle: *the model proposes; deterministic code disposes.* No
step trusts the previous one; every step fails **closed** (deny/escalate, never
silently proceed).

| # | Step | Guardrails (input → output) | On violation |
|--:|------|------------------------------|--------------|
| 1 | **Ingress / gateway** | authN (JWT); token-bucket rate limit; request-size + payload schema validation; channel/identity verification | `401` / `429` / `400`; drop |
| 2 | **PII masking** | mandatory mask/tokenize **before any model sees text** (NER + deny-list); post-check that no raw PII passes downstream | **fail-closed** — block the turn, alert; never send unmasked to an LLM |
| 3 | **Prompt-injection defense** | treat customer text as data, not instructions; strip/ignore embedded directives; per-agent tool allow-list; system-prompt integrity check | ignore injected instructions; writes still gated downstream regardless of model output |
| 4 | **Triage / intent** | confidence ≥ threshold; intent ∈ allowed set; assign risk tier | `< threshold` → ask a clarifying question or route to human; unknown intent → human |
| 5 | **RAG retrieval** | relevance score ≥ threshold; **policy version pinning**; citations required; grounded-only answering | **no supporting doc → don't answer** (say so / escalate), never fabricate policy |
| 6 | **Planning (LLM step)** | hard **N=5** step cap; semantic loop detection; output-schema validation; action ∈ allowed set | interrupt + escalate; repair/retry within budget |
| 7 | **Read-only tool call** | per-agent tool allow-list; arg schema validation; **scope check — customer may access only their own accounts/txns** | reject call; block cross-account access (security-critical) |
| 8 | **Write / action proposal** | **Policy & Guardrails Engine**: amount ≤ policy limit; ≤ auto-approve threshold; eligibility; legality; duplicate check | **block the API call** + escalate to HITL |
| 9 | **Idempotency** | orchestrator-issued key required on every external write; dedup store lookup | return prior result — no double charge/refund/pull |
| 10 | **External dependency call** | circuit breaker; timeout; bounded retry w/ backoff; response validation | open circuit → fallback / **park the case**, don't fail it |
| 11 | **Domain decision (dispute/loan)** | **decision rules in code, not the model**; regulatory checks (Reg E eligibility; ECOA); a decline **requires** an adverse-action notice + human sign-off | refer to human; block any auto-decline lacking a notice |
| 12 | **HITL gate** | risk tier ≥ 2, or confidence < 75%, or amount > threshold ⇒ mandatory approval | durable wait; **no autonomous write** proceeds |
| 13 | **Response generation** | output PII scrub; **no unauthorized financial/legal advice**; required disclosures present; tone/compliance rubric | regenerate/redact; fall back to a vetted template |
| 14 | **Notification / side effect** | verified recipient; channel **consent**; idempotent send | suppress + log |
| 15 | **Audit** | **no state-changing action commits without an audit record** | **fail-closed** — if the audit write fails, the action fails |
| 16 | **Workflow (case-level)** | durable checkpoint per step; regulatory **SLA timers**; per-case **budget** (wall-clock, steps, tokens, $) | timer fires (e.g., auto provisional credit); budget exceeded → escalate |
| 17 | **Global kill switch** | control plane can halt all autonomous writes instantly (scope: global / tool / tenant) | every write routes to the human queue |

**Layering:** guards 1–3 protect the boundary, 4–7 bound what the agent *reads*
and reasons about, 8–12 bound what it *does*, 13–14 bound what leaves the system,
and 15–17 are the always-on safety net (audit, durability, kill switch). A failure
at any layer degrades to the next-safer path — the worst case is "a human handles
it," never "an unchecked action ships."

---

## Data Model

```sql
-- Durable case (source of truth); the conversation is ephemeral around it.
CREATE TABLE cases (
    id            UUID PRIMARY KEY,
    conversation_id UUID NOT NULL,
    customer_id   UUID NOT NULL,
    type          TEXT NOT NULL,          -- dispute | loan | ...
    status        TEXT NOT NULL,          -- open | awaiting_approval | awaiting_network | resolved | declined
    risk_tier     SMALLINT NOT NULL,      -- 1 autonomous ... 3 human-required
    workflow_id   TEXT NOT NULL,          -- Temporal handle
    created_at    TIMESTAMPTZ DEFAULT now(),
    updated_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE disputes (
    case_id            UUID PRIMARY KEY REFERENCES cases(id),
    txn_id             TEXT NOT NULL,
    amount_cents       BIGINT NOT NULL,
    reason_code        TEXT NOT NULL,
    network_ref        TEXT,
    provisional_credit BOOLEAN DEFAULT false,
    reg_e_deadline     TIMESTAMPTZ NOT NULL,   -- regulatory SLA timer
    status             TEXT NOT NULL
);

CREATE TABLE loan_applications (
    case_id           UUID PRIMARY KEY REFERENCES cases(id),
    product           TEXT NOT NULL,
    amount_cents      BIGINT NOT NULL,
    applicant_details JSONB NOT NULL,          -- stated data; SSN encrypted AES-256-GCM
    bureau_ref        TEXT,
    decision          TEXT,                    -- approved | referred | declined
    adverse_action_ref UUID                    -- notice, if declined
);

-- Idempotency for every external write.
CREATE TABLE idempotency_keys (
    key          TEXT PRIMARY KEY,
    case_id      UUID NOT NULL,
    tool         TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    response     JSONB,
    status       TEXT NOT NULL,               -- in_flight | done | failed
    created_at   TIMESTAMPTZ DEFAULT now()
);

-- Immutable, append-only audit (no UPDATE/DELETE grants).
CREATE TABLE audit_log (
    id        BIGSERIAL PRIMARY KEY,
    actor     TEXT NOT NULL,                  -- agent:<id> | human:<id> | system
    action    TEXT NOT NULL,
    target    TEXT NOT NULL,
    before    JSONB,
    after     JSONB,
    reason    TEXT,
    ts        TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE policy_docs (
    id        UUID PRIMARY KEY,
    version   INT NOT NULL,
    text      TEXT NOT NULL,
    embedding VECTOR(1024)
);
```

**Indexes:** `cases(customer_id, status)`, `cases(status, updated_at)` (ops
queues), `disputes(reg_e_deadline)` (SLA sweeps), `audit_log(target, ts)`,
`policy_docs` HNSW on `embedding`.

**Partitioning:** `audit_log` and `messages` partitioned by month (append-heavy,
time-queried). Everything else fits comfortably at this scale — no sharding needed.

---

## Infrastructure Choices

| Component | Choice | Notes |
|-----------|--------|-------|
| Database | **PostgreSQL** (+ pgvector) | Cases, decisions, audit — ACID; RAG embeddings in the same store. |
| Message Queue | **Kafka** | Case/audit event backbone; fan-out to CRM, notifications, analytics, audit sink. |
| Cache / working memory | **Redis** | Conversation working memory, idempotency TTL, rate-limit counters. |
| Workflow Engine | **Temporal** | Durable multi-step cases, HITL signal waits, Reg E SLA timers that survive restarts. |
| Rate Limiter | **Token bucket** (Redis) | Per customer/API key at the gateway; bursts OK. |
| Circuit Breaker | **Per external dependency** | Card network, bureau, KYC, doc processor, LLM — independent breakers. |
| Models | **Tiered** | Triage/drafting → small/cheap; dispute/loan reasoning + tool orchestration → large. |

**Database — PostgreSQL:** relational, ACID (needed for money + audit),
pgvector for policy RAG in one system. Scale is modest (50k/day) — no need for a
wide-column store. **Partitioning:** time-based on `audit_log`/`messages`.

**Message Queue — Kafka:** durable, replayable event log; one case update fans out
to notifier, CRM, analytics, and the audit sink. *(At this volume SQS would also
suffice; Kafka is chosen for replay + multi-consumer fan-out and an
append-only event history that complements the audit log.)*
**Backpressure:** consumers are idempotent and commit offsets after processing;
lag is a monitored alert, not data loss.

**Redis:** working memory (conversation state, active goal), idempotency keys
(TTL 24h), rate-limit counters (TTL = window). **Not** a source of truth — cases,
decisions, and audit never live only in Redis.

**Workflow Engine — Temporal:** the crux for the case plane. Cases span minutes to
days, each step can fail independently, and several **wait on external events or
human approval**. Temporal gives durable step-level checkpointing, retries,
**signals** (HITL / network webhooks) and **timers** (Reg E provisional-credit
deadline) that survive worker restarts — none of which we want to hand-build.
Celery/SQS would force us to reinvent durable waits and replay.

**Rate Limiter — token bucket:** per customer/API key (primary), per IP for
unauthenticated endpoints, tighter per-endpoint limits on expensive ops (doc
upload). `429` + `Retry-After` on exceed.

**Circuit Breaker:** one per external integration; on open, route to the fallback
(queue the case for retry / underwriter, or degrade to "we're on it" instead of
surfacing a raw error). A degraded bureau must not exhaust workers.

---

## Test & Evaluation Framework

| Layer | What it covers | Tooling |
|-------|---------------|---------|
| Unit | Guardrail rules, routing, risk-tiering, schemas | pytest |
| Integration | Temporal workflows (incl. HITL signal + timer paths), DB, queue | testcontainers + Temporal test env |
| Contract | External API + webhook shapes (network, bureau) | Pact / OpenAPI + recorded fixtures |
| Load | Fast-track p95 under 5× spike; workflow throughput | Locust / k6 |
| Eval (AI) | Intent accuracy, tool-selection correctness, guardrail catch-rate, confidence calibration | LLM-as-judge + golden ticket set |

**Regression gate for model upgrades:** replay a **golden set of curated tickets**
(disputes, loans, edge cases) through the full pipeline; block deploy on any shift
in decision boundaries, guardrail catch-rate, or escalation rate beyond tolerance.
Guardrails are tested independently of the model so a model swap can't silently
loosen them. Full detail below.

---

## Evaluation Pipeline

Agentic systems fail *between* the steps, so we evaluate **every step**, not just
the final answer. Three levels, run both **offline** (pre-deploy gate) and
**online** (in production):

1. **Component evals** — each step in isolation (was the intent right? was the
   retrieval grounded? did the guardrail catch the bad action?).
2. **Trajectory evals** — the *path* the agent took (right tools, right order, no
   wasted or looping steps) vs. golden traces.
3. **Outcome evals** — the case result (correctly resolved? correctly escalated?
   compliant? within cost/latency?).

Every `AgentStep` is already traced/logged (see Observability); evals **replay**
those traces offline and **sample** them online — evaluation *is* pipeline
instrumentation, not a separate output check.

### Per-step evaluation

| Step | What we measure | Method | Gate (block deploy if…) |
|------|-----------------|--------|--------------------------|
| Triage / intent | intent accuracy, risk-tier accuracy | labeled set + confusion matrix | F1 < target, or **any** tier-3 mislabeled as tier-1 (under-tiering is unsafe) |
| RAG retrieval | recall@k, precision, **faithfulness/groundedness**, citation correctness | golden Q→doc set + LLM-judge faithfulness | recall@k or faithfulness below target |
| Tool selection | correct tool + correct args | trajectory eval vs. golden traces; schema pass-rate | tool-choice accuracy or arg-validity drop |
| Reasoning/plan | step efficiency (steps-to-resolution), loop rate | trajectory metrics | efficiency regresses or loop rate rises |
| **Guardrail engine** | catch-rate on invalid/illegal actions; false-block rate | **red-team action set** | catch-rate < 99% on the illegal set, or false-blocks over budget |
| Domain decision (dispute/loan) | decision accuracy vs. ground truth / underwriter labels | golden decisions + human adjudication | accuracy below target or **fairness** breach (below) |
| HITL routing | escalation precision & recall | labeled edge set | over-escalation (kills autonomy) or under-escalation (unsafe) |
| Response | correctness, helpfulness, tone, **compliance** (no unauthorized advice, disclosures present) | LLM-judge rubric + human sample | quality/compliance below target |
| Confidence calibration | is "75%" really 75%? | **ECE** (expected calibration error) | ECE above threshold (the HITL cutoff must mean something) |
| Safety | **PII-leak rate**, prompt-injection resistance, cross-account access | adversarial suite | PII leak **> 0**, or any successful injection/cross-account probe |
| End-to-end | auto-resolution rate, containment, time-to-resolution, **cost/ticket**, CSAT | full-pipeline replay + online metrics | resolution/cost/latency past target |

### Datasets

- **Golden set** — curated, anonymized real tickets per domain + edge cases, each
  labeled with expected intent, ideal trajectory, and expected decision. The
  backbone of offline eval and the regression gate.
- **Adversarial / red-team set** — prompt injections, jailbreaks, out-of-scope
  asks, illegal-action attempts, cross-account probes, hallucination bait. Grows
  every time an incident is found.
- **Fairness set** — matched loan applications differing *only* on protected
  attributes, to test ECOA disparate impact.
- **Synthetic augmentation** — LLM-generated variants for coverage of rare
  intents/edge cases, human-spot-checked before entering the golden set.

### Methods

- **Deterministic checks** (exact): schema validity, guardrail catch, PII regex,
  citation presence, scope violations. Fast, unambiguous — the CI floor.
- **LLM-as-judge** (faithfulness, helpfulness, tone, compliance): rubric-based,
  **human-calibrated**, multiple judges/lenses to reduce judge bias; judge itself
  is periodically validated against human labels.
- **Trajectory eval**: compare the tool-call sequence to golden traces; score step
  efficiency and detect loops.
- **Statistical**: calibration (ECE); **fairness** (disparate-impact ratio / 80%
  rule) on the fairness set.
- **Human review**: sampled, weighted toward decisions and escalations; every
  human override becomes a new golden case.

### Offline (pre-deploy regression gate)

On every model / prompt / policy change, replay the **golden + adversarial +
fairness** sets through the *real* pipeline. **Block the deploy** on any of:

- decision-boundary shift beyond tolerance,
- guardrail catch-rate drop, or false-block spike,
- escalation-rate drift (either direction),
- retrieval faithfulness / response-quality regression,
- calibration (ECE) regression,
- **fairness** threshold breach,
- **PII-leak > 0** (hard stop),
- projected cost/ticket over the $0.08 cap.

Guardrails are evaluated **independently of the model**, so a model swap can never
silently loosen them.

### Online (production)

- **Shadow mode** — a new model/policy runs in parallel on live traffic with **no
  customer-visible effect**; compare its decisions/trajectories to the incumbent
  before promoting.
- **Canary / A-B** — gradual rollout gated on live guardrail + outcome metrics;
  auto-rollback on regression.
- **Live sampling → human review** — auto-flag every low-confidence turn and every
  guardrail block; sample a % of auto-resolutions for audit.
- **Continuous metrics** — resolution rate, escalation drift, cost/ticket, CSAT,
  and **PII-leak = 0** as a standing alarm.
- **Feedback loop** — human overrides + customer 👍/👎 flow back into the golden
  and adversarial sets, so coverage compounds over time.

### Cadence & ownership

- **CI gate** on every model/prompt/policy PR (component + guardrail + safety subset).
- **Nightly** full golden + adversarial replay.
- **Weekly** fairness + calibration audit.
- **Continuous** online metrics + sampled human review.

---

## Observability / Debuggability / Telemetry

**Metrics to alert on:**
- p95/p99 latency — per stage (triage, RAG, tool exec) and per model tier.
- Error rate by class: validation, external dependency, model/malformed, guardrail-block.
- **Cost/tokens per ticket** vs the $0.08 cap (alert on drift).
- **Escalation rate** and **auto-resolution rate** (both drifting = model or policy change).
- Queue depth / consumer lag; open circuit breakers; Reg E deadlines approaching.

**Tracing:** one distributed trace per conversation *and* per case; spans for every
agent step (thought → tool call → guardrail → tool result), so a slow/expensive/
wrong run is inspectable end-to-end (Logfire/OpenTelemetry over the agent loop).

**Logging:** structured JSON; every line carries `conversation_id`, `case_id`,
`step_n`, `actor`, `tokens`, `latency_ms`, and **never raw PII** (masked upstream).

**Runbook hooks:** on-call first looks at the control-plane dashboard — open
circuits, escalation-rate spike, cost spike, stuck workflows — and can trip the
**kill switch** or tighten thresholds without a deploy.

---

## Deep Dives

### Deep Dive 1: Two-plane architecture & why not L5 model-orchestration

The defining decision. A **conversation** (fast, turn-based, ephemeral) and a
**case** (durable, multi-day, money-touching) have opposite requirements, so they
get different runtimes:

- **Conversational plane** — stateless workers + Redis working memory; optimized
  for low latency and streaming; handles dialogue and the read-only fast track.
- **Case plane** — Temporal workflows; optimized for durability, retries, timers,
  and human waits; handles disputes/loans.

Routing between them and orchestration *within* the case are **code** (L2 + the
workflow), not a reasoning LLM (L5). Why: an L5 orchestrator that decides
delegation dynamically is non-deterministic and hard to audit — unacceptable when
a regulator can ask "why did the system take this action?" The model's autonomy is
**scoped inside** each L3/L4 agent's bounded tool loop; the *control flow* is
deterministic and logged. This is the bounded-autonomy stance made concrete.

### Deep Dive 2: The agentic control loop & deterministic guardrails

Each agent turn runs `PLANNING → TOOL EXECUTION → OBSERVATION → ASSESSMENT` with
two hard safety mechanisms:

1. **Bounded reasoning:** a hard cap of **N = 5** steps per turn, plus
   **semantic-similarity loop detection** — if consecutive "thoughts" are
   near-identical (no progress), the loop is interrupted and the case escalates.
2. **Guardrails Engine in front of every write:** tool calls are proposals, not
   commands. The engine deterministically checks amount ≤ policy limit, action ≤
   auto-approve threshold, eligibility, and legality *before* the API is touched.
   A refund exceeding the disputed amount, or a loan approval outside limits, is
   **blocked and escalated** — the model cannot override it.

Combined with **idempotency keys** on every external write, a crash-and-retry or a
model mistake cannot double-move money.

### Deep Dive 3: HITL, durable waits & the operator control plane

Two ways a human enters, both **durable**:

- **In-case approval:** an irreversible/low-confidence step raises an
  `ApprovalTask` and the Temporal workflow **waits on a signal** — for minutes or
  days — surviving worker restarts. On decision, it resumes with the human's input
  in context. Regulatory timers (Reg E provisional credit; loan SLA) run as
  workflow timers in parallel, so a deadline fires even while waiting.
- **Operator control plane** (separate, mTLS+RBAC, fully audited): **global kill
  switch** (halt autonomy instantly, no deploy), **runtime thresholds**
  (auto-approve limits, confidence cutoffs — config, not code), **review/override
  queue** (all escalations + a sampled % of auto-resolutions), and
  **per-tool/per-tenant toggles** (disable autonomous refunds during an incident,
  keep read-only lookups live). The agent can never call this plane.

### Deep Dive 4: Cost & latency control (hitting $0.08/ticket and p95 < 6 s)

- **Fast-track first:** a cheap classifier resolves/acknowledges simple intents in
  ~1 s; the large model is invoked only for policy interpretation and dispute/loan
  reasoning. With the bulk of tickets on the cheap path, the blended average stays
  near $0.08 even though a complex case costs more.
- **Context Window Manager:** long multi-turn conversations are summarized/
  compressed ("summarized context") so active tokens stay bounded — controlling
  both latency and per-turn cost (context re-sent every call is the dominant cost
  driver).
- **Tiered models** end-to-end (triage/drafting cheap; reasoning large), the same
  cost lever proven in `basics/pydantic_ai/agent_complexity`.

---

## Fault Analysis / Edge Cases

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Card network / bureau down | Case can't progress | Circuit breaker → durable retry with backoff; case parks, doesn't fail; customer told "in progress". |
| Worker crashes mid-refund | Risk of double or lost action | Temporal replay from last checkpoint + **idempotency key** → completed side effects not repeated. |
| Model returns malformed / illegal action | Bad or unsafe action | Schema validation + **Guardrails Engine** block + retry budget → escalate. |
| Reasoning loop stalls | Latency + token burn | N=5 cap + semantic loop detection → interrupt + escalate. |
| Duplicate customer message (retry) | Two cases opened | **Idempotency-Key** on message POST. |
| Reg E deadline approaching, still waiting on network | Compliance breach | Workflow **timer** auto-issues provisional credit within the window. |
| Confidence low / ambiguous intent | Wrong autonomous action | Confidence < 75% → HITL before any write. |
| LLM provider outage | Agent can't reason | LLM circuit breaker → fall back to templated flows + human queue for complex cases; simple lookups (non-LLM tools) still serve. |
| PII leak risk | Regulatory breach | Masking at the edge *before* the LLM; logs carry masked data only. |
| Prompt-injection via customer text | Agent tricked into bad action | Guardrails are code (not promptable); writes gated regardless of model output; tool allow-list per agent. |

---

## Tradeoffs Summary

| Decision | Chosen | Alternative | Why |
|----------|--------|-------------|-----|
| Orchestration control | **Code (L2 router + Temporal)** | L5 model-orchestrator | Determinism + auditability are regulatory requirements. |
| Two planes | **Separate conversation vs case** | One loop for everything | Opposite latency/durability needs; clean sync-vs-async split. |
| Autonomy | **Bounded + HITL + kill switch** | Full autonomy | Irreversible financial actions are a liability; compliance forbids it. |
| Workflow engine | **Temporal** | Celery/SQS + hand-rolled state | Durable waits, timers, signals, replay out of the box. |
| Guardrails | **Deterministic engine before writes** | Trust the model + prompt rules | Model output is a proposal; money movement needs hard rules. |
| Complex cases | **Async durable workflow** | Block the request | Disputes/loans span days; can't hold a connection. |
| Store | **Postgres + pgvector** | Separate vector DB / NoSQL | ACID for money/audit; RAG in one system; scale doesn't need more. |
| Cost | **Fast-track + tiered models** | One large model everywhere | Keeps blended cost near $0.08 without gutting quality. |
