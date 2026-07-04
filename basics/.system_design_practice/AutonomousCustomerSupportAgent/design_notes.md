# Autonomous Customer Support Agent — Design Notes

> Raw design notes, formatted and lightly corrected for the financial-support
> prompt. Not the final design doc — see [Corrections applied](#corrections-applied)
> at the end for what changed and why. Next step: expand into a full `Design.md`
> using `../sd_template.md`.

## Table of Contents

- [Design prompt](#design-prompt)
- [Key design patterns to cover](#key-design-patterns-to-cover)
- [1. Scope & Goals](#1-scope--goals)
- [2. Constraints & Guardrails](#2-constraints--guardrails)
- [3. High-Level Architecture](#3-high-level-architecture)
- [4. Deep-Dive Components](#4-deep-dive-components)
- [5. Trade-offs & Bottlenecks](#5-trade-offs--bottlenecks)
- [6. Observability & Evaluation](#6-observability--evaluation)
- [Corrections applied](#corrections-applied)

## Design prompt

Design an autonomous customer support agent capable of resolving complex,
multi-step financial inquiries (e.g., disputed charges or loan applications)
without human intervention.

> **Design stance — bounded / delegated autonomy, not full autonomy.**
> "Without human intervention" means *no human in the common path*, not *never*.
> The agent resolves the majority of cases end-to-end; humans are reserved for
> irreversible or low-confidence decisions (see HITL) and retain override control
> at all times. **In banking, full autonomy over irreversible actions is a
> liability, not a feature** — it is constrained by regulation (ECOA / fair-lending
> and adverse-action notices for loans; Reg E / Reg Z error-resolution rules for
> disputes; SR 11-7 model-risk governance) and by security/fraud risk. Industry
> practice for banking support AI is deliberately partial autonomy behind strict
> guardrails, an operator control plane, and human fail-safes. This design follows
> that posture.

## Key design patterns to cover

- **Control Loop Architecture:** a central orchestrator running the
  Observe → Think → Act loop.
- **State Machines & Directed Graphs:** managing the state of a multi-turn
  conversation and deciding when an agent hands off control to another agent or
  to a human operator.
- **Deterministic Fallbacks:** strict rules that override the model when it
  hallucinates or attempts an invalid or illegal action.

## 1. Scope & Goals

- **Objective:** an autonomous customer support agent that resolves complex,
  multi-step financial inquiries (disputed charges, loan applications) without a
  human in the common path.
- **Autonomy level — bounded / delegated (partial), with human oversight.**
  Autonomy is *high within a safe envelope*, deliberately capped below full
  autonomy for irreversible/regulated actions. The agent performs tool execution
  autonomously for **tier-1** cases (balance / transaction-status inquiries,
  in-policy refunds or credits under a threshold, first-pass dispute intake,
  document collection for a loan). **Tier-2 / edge cases** (large or ambiguous
  disputes, loan underwriting/approval, suspected fraud) are escalated to humans
  with a fully populated context. Operators can pause, override, or reclassify at
  any time (see Admin control & fail-safes).
- **Scale:** 50,000 tickets/day (~0.6 tickets/s average, ~2–3/s peak), with
  capacity to absorb a 5× spike to 250,000 tickets/day (~3/s average, ~10–15/s
  peak).
- **Quality SLA:**
  - *Simple, synchronous queries* (balance, transaction status, policy Q&A):
    **p95 < 6 s**.
  - *Complex, multi-step cases* (disputed charges, loan applications): handled
    **asynchronously as durable workflows**. Resolution can span seconds to days
    (a chargeback lifecycle or a credit-bureau pull is inherently slow), so the
    SLA is measured on **time-to-first-response** and **time-to-resolution**, not
    single-request latency.
- **Cost cap:** average **$0.08 per ticket**. Achievable only because the
  fast-track path handles the bulk of tickets cheaply; complex cases cost more
  and are the minority (see Trade-off 1).

## 2. Constraints & Guardrails

- **Security & Compliance:** PII (names, addresses, account/card numbers, SSNs,
  transaction IDs) must be **masked/tokenized before reaching the LLM** and
  scrubbed from conversation logs. Every autonomous action and decision is written
  to an **immutable audit log** (regulatory requirement for financial operations —
  who/what/why for each decision). Applicable regimes constrain what may be
  automated: **ECOA / fair-lending + adverse-action notices** (loans),
  **Reg E / Reg Z** error-resolution timelines (disputes), **SR 11-7** model-risk
  governance. These are *why* the autonomy is bounded, not merely how.
- **Admin control & fail-safes (operator control plane):** a human-operable
  console that sits above the agent, so the bank always retains control:
  - **Global kill switch** — instantly halt all autonomous actions (fall back to
    human queues) without a deploy.
  - **Ops-configurable risk thresholds** — auto-approve limits, confidence cutoffs,
    and per-action-type autonomy toggles are runtime config, not hard-coded.
  - **Review & override queue** — every escalation (and a sampled % of autonomous
    resolutions) is reviewable; operators can override or reverse.
  - **Per-tool / per-tenant enable-disable** — e.g., disable autonomous refunds
    during an incident while leaving read-only lookups on.
- **State management:** agent execution must be **durable**. If an orchestration
  worker crashes midway through a multi-step refund/API sequence, state is
  recoverable and the workflow resumes **without re-executing completed side
  effects**.
- **Autonomy boundaries:** actions are classified by risk. **Read-only** actions
  are autonomous; **irreversible / financial-impact** actions (final refunds or
  credits above a threshold, loan approvals, account changes) require
  **human-in-the-loop (HITL) approval**.

## 3. High-Level Architecture

The system is built on an **observe → think → act → observe** loop driven by a
**stateless orchestration service** that reads/writes to external state storage.

- **Intake layer:** web / chat / email / IVR channels → API Gateway with
  authentication and rate limiting.
- **Router agent:** analyzes customer intent and routes to domain sub-agents
  (**Disputes**, **Loans**, **Payments/Refunds**, **Account**, **Policy/RAG**).
- **State orchestrator:** manages the agentic loop, handles retries and timeouts,
  and owns durable workflow state.
- **Memory systems:**
  - **Working memory:** conversation history and active goal state, in a fast
    key-value store (Redis).
  - **Semantic memory:** vector database (PostgreSQL + pgvector) for
    Retrieval-Augmented Generation (RAG) over company policies, T&Cs, and
    regulatory rules.
  - **Account / long-term memory:** customer profile, account state, and prior
    ticket history from the system-of-record DB (source of truth for financial
    data — never the vector store).

## 4. Deep-Dive Components

### A. The Agentic Loop

The orchestrator drives an execution state machine:
`[PLANNING] → [TOOL EXECUTION] → [OBSERVATION] → [ASSESSMENT]`.

To prevent infinite reasoning loops, the orchestrator sets a **hard limit of
N = 5** reasoning/tool-calling steps per turn. **Semantic-similarity tracking** of
recent thoughts programmatically interrupts the loop if the agent is stuck
(repeating near-identical reasoning without progress).

### B. Tooling Layer

Tools expose **standardized JSON schemas**. Every tool execution passes through a
**Policy & Guardrails Engine** before the call is made.

- **Example guardrail (financial):** if the agent proposes a refund/credit that
  **exceeds the disputed transaction amount**, or exceeds the **auto-approve
  limit**, the policy engine **blocks the API call** and triggers an automatic
  escalation to a human agent.
- **Idempotency:** all write APIs (payment processor, core-banking ledger,
  card-network dispute API) include **idempotency keys generated by the
  orchestrator**, so a failed or retried tool execution never double-charges or
  double-refunds.

### C. Human-in-the-Loop (HITL)

For irreversible actions or low-confidence assessments (e.g., intent confidence
**below a 75% threshold**):

1. The agent transitions to **`WAIT_FOR_APPROVAL`**.
2. State is serialized and a ticket is pushed to an internal CRM / task queue.
3. A human agent approves or rejects.
4. The orchestrator **resumes** the agent from the saved state, passing the human
   response back into the context window.

## 5. Trade-offs & Bottlenecks

### Trade-off 1 — Latency vs. Reasoning Depth

- **Problem:** multi-step reasoning chains and tool calls can easily push response
  times past 20 seconds.
- **Solution:** a **Fast-Track Classifier** (smaller, cheaper model) immediately
  handles/acknowledges simple queries — the financial analog of "where is my
  order" is *balance / transaction-status / recent-activity* lookups — within
  ~1 second. Invoke the large model only for policy interpretation and dispute /
  loan reasoning.

### Trade-off 2 — Statefulness vs. Scalability

- **Problem:** keeping an entire agentic conversation in context inflates token
  cost over long, multi-turn interactions.
- **Solution:** **stateless execution layer + external state storage**. A
  **Context Window Manager** summarizes/compresses old turns ("summarized
  context") to keep the active token count below threshold.

## 6. Observability & Evaluation

Evaluating agentic AI requires **pipeline instrumentation**, not just output
checks.

- **Pipeline telemetry:** trace every node (intent recognition → RAG retrieval →
  tool generation → guardrail check → tool execution) to localize failures.
- **Cost tracking:** log input/output token usage per ticket to monitor adherence
  to the $0.08 cap.
- **Evaluation:** offline evaluation on **golden datasets** (curated tickets)
  benchmarks model updates and guardrail logic **before deployment**; used as a
  **regression gate** to catch decision-boundary shifts on model upgrades.

---

## Corrections applied

Substantive changes made while formatting (for review):

1. **Domain realignment (e-commerce → financial).** The prompt is about financial
   inquiries, but several examples were retail:
   - Sub-agents `Returns, Policy` → `Disputes, Loans, Payments/Refunds, Account,
     Policy/RAG`.
   - Guardrail example "$500 refund for a $50 item" → "refund/credit exceeding the
     disputed transaction amount or the auto-approve limit."
   - Idempotency example `Stripe, Shopify` → `payment processor, core-banking
     ledger, card-network dispute API`.
   - Fast-track example `WISMO` (where-is-my-order) → balance / transaction-status
     / recent-activity lookups.
2. **Latency SLA clarified.** "p95 < 6 s" applies to *simple synchronous* queries.
   Complex disputes/loans are inherently long (chargeback lifecycle, bureau pulls)
   → framed as **async durable workflows** with time-to-first-response /
   time-to-resolution SLAs instead.
3. **Confidence-threshold wording fixed.** "intent not recognized > 75%" was
   ambiguous/backwards → escalate when **confidence is *below* 75%**.
4. **"Without human intervention" framing.** Added an explicit interpretation
   (no human in the common path, not never) to reconcile the prompt with the HITL
   design — irreversible financial actions can't be fully human-free.
5. **Added: immutable audit log** of every autonomous decision (compliance
   necessity for financial ops).
6. **Added: account / long-term memory** tier — financial resolution needs
   account state + prior-ticket history, kept in the system-of-record DB, not the
   vector store.
7. **Added capacity arithmetic** (~0.6/s avg, ~3/s at 5× spike) so the scale
   number is actionable.
8. **Formatting:** converted LaTeX arrows (`$\rightarrow$`) to `→`, normalized
   headings, lists, and added a TOC.
9. **Reframed autonomy as bounded / delegated (partial), not "High."** Banking
   forbids full autonomy over irreversible actions; the design stance now names
   this explicitly and ties it to the governing regulations (ECOA/adverse-action,
   Reg E/Z, SR 11-7). *(per review — this is a partial-autonomy + HITL-override
   system, matching standard industry posture.)*
10. **Added an Admin control & fail-safes (operator control plane)** guardrail —
    global kill switch, ops-configurable thresholds, review/override queue,
    per-tool/per-tenant toggles — so the bank retains control at all times.
