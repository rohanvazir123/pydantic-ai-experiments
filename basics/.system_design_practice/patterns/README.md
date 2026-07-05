# Patterns

Reusable system-design pattern notes — reference material shared across the
designs in `.system_design_practice/` (LoanApproval, AutonomousCustomerSupportAgent,
…). Each file is a condensed, interview-ready write-up: what the pattern is, how
it works, and the trade-offs with justifications.

## Table of Contents

- [Contents](#contents)
- [Conventions](#conventions)

## Contents

| Pattern | File | One-liner |
|---------|------|-----------|
| Transactional Outbox | [`transactional_outbox.md`](transactional_outbox.md) | Atomic state-change + event emission without 2PC; DB outbox row + CDC/Debezium relay to Kafka. |

## Conventions

- One pattern per file; each starts with a **Verdict / TL;DR**, then problem → how
  it works → trade-offs (gains and costs, each justified) → when to use/not →
  related patterns.
- Kept implementation-flavored but vendor-neutral where it matters.
- Cross-reference the concrete designs that use the pattern.
