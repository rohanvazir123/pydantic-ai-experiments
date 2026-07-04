# Session Context

## What we're working on

System design interview prep. Two problems being designed and deep-dived:

1. **Multi-Step Risk-Aware Loan Application Router** — `LoanApproval/Design.md`
2. **Automated Transaction Dispute Pipeline** — not yet started (next)

---

## Current state of LoanApproval/Design.md

First cut is complete. Sections done:
- Problem statement, functional + non-functional requirements
- Capacity estimation (1M applications/day, ~12 avg req/s, 60–120 peak)
- Core entities including `LoanApplicationDetails` (raw applicant-submitted PII)
- API design (REST, PUT over PATCH, idempotency header)
- Infrastructure choices: PostgreSQL, Kafka, Redis, Temporal, token bucket rate limiter, circuit breaker per dependency
- High-level architecture diagram
- Data flow (Temporal workflow with parallel activities + HIL signal/timer pattern)
- Agentic AI components (L1 for verification steps, L3 for risk synthesis, code for routing)
- Data model with SQL (including `loan_application_details` with encrypted SSN)
- Test & eval framework
- Observability
- Deep dives: idempotency for credit pulls, regulatory rules engine, worker pool scaling, Temporal workflow design (HIL + SLA timer)
- Fault analysis
- Tradeoffs summary

## Key design decisions made

- **Temporal** chosen as workflow engine — multi-step durable execution, HIL waits via signals, SLA timers that survive worker restarts
- **L2 outer loop** — code controls flow/routing; model handles interpretation only (regulatory requirement)
- **No partial results** — if any verification step fails after retries, route to underwriter, never auto-decide
- **LoanApplicationDetails** — raw stated data (SSN encrypted AES-256-GCM, stated income, employer) separate from verification results; risk agent compares stated vs verified
- **PUT over PATCH** everywhere retries matter
- **Circuit breaker per external dependency** (bureau, ID provider, doc processor)

## Next session agenda

User will drill on both designs — questions about failure handling, evaluation,
model upgrade safety, latency, regulatory edge cases. Then we go through together.

Transaction Dispute Pipeline design still to be started — use `sd_template.md` as
the base, same approach as LoanApproval.

## Files to know about

- `.system_design_practice/LoanApproval/Design.md` — loan approval system design
- `.system_design_practice/LoanApproval/agentic_banking_pipelines.md` — original blueprint for both problems
- `.system_design_practice/sd_template.md` — standard template for all future designs
- `basics/workflows/temporal_python_sdk.md` — Temporal Python SDK reference
