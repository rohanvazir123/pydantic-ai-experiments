# Pattern: Transactional Outbox (with CDC / Debezium / Kafka)

## Table of Contents

- [Verdict](#verdict)
- [Problem it solves: the dual-write problem](#problem-it-solves-the-dual-write-problem)
- [What the pattern is, and where Debezium/Kafka fit](#what-the-pattern-is-and-where-debeziumkafka-fit)
- [How it works (end to end)](#how-it-works-end-to-end)
- [Trade-offs](#trade-offs)
- [When to use / when not](#when-to-use--when-not)
- [Related patterns](#related-patterns)

## Verdict

`DB + CDC + Debezium + Kafka = Transactional Outbox` is **directionally right but
imprecise**. Precisely:

> **Outbox = (business write + outbox row in ONE local transaction) + a message
> relay.** Debezium + Kafka is the *log-tailing relay* — and it's the canonical
> Outbox pattern **only with a dedicated outbox table** (ideally + Debezium's
> Outbox Event Router). Point Debezium straight at business tables and you have
> *CDC-based event publishing* ("listen-to-yourself"), which fixes the same
> dual-write problem but emits row-level CRUD deltas, not domain events.

## Problem it solves: the dual-write problem

A service must do two things when something happens: (1) persist state, (2) publish
an event. They're two systems with **no shared transaction**, so failures diverge:

- DB commits, Kafka publish fails → state changed, **no event**.
- Kafka publish succeeds, DB rolls back → **event for a thing that doesn't exist**.

The naive fix — 2PC/XA across DB + Kafka — is slow, heavy, and Kafka doesn't do XA
well. Outbox is the pragmatic alternative.

## What the pattern is, and where Debezium/Kafka fit

Two parts:

1. **Write side (the actual pattern):** in one **ACID transaction**, write the
   business rows *and* an `outbox` row (the event payload). Atomic — both or
   neither. Needs nothing but your DB. This is what kills the dual-write problem.
2. **Relay side (message relay):** moves outbox rows to Kafka. Two builds:
   - **Polling publisher** — `SELECT` unsent rows, publish, mark sent. Simple;
     polls the DB, adds latency.
   - **Log tailing / CDC** — read the DB commit log (Postgres WAL via logical
     decoding). **This is Debezium**, streaming via Kafka Connect.

| Term | Role |
|------|------|
| DB (+ outbox table) | Write side — atomicity. *This is the pattern.* |
| CDC / Debezium | The relay — log-tailing message relay. *Implementation choice.* |
| Kafka | Transport the relay publishes to. |

The canonical "Outbox via CDC" adds the **Debezium Outbox Event Router** (SMT):
maps outbox columns → topic (`aggregatetype`), Kafka key (`aggregateid`), value
(`payload`) → clean domain events with a stable contract, no internal-schema leak.

## How it works (end to end)

```
BEGIN;
  INSERT INTO orders (...);                      -- business state
  INSERT INTO outbox (aggregate_type, aggregate_id, type, payload) VALUES (...);
COMMIT;                                          -- atomic: both or neither
      |
      v  Postgres writes both to the WAL
Debezium tails the WAL (logical replication slot)
      |  Outbox Event Router: row -> topic + key(aggregate_id) + payload
      v
Kafka topic (partitioned by aggregate_id -> per-aggregate ordering)
      |  at-least-once
      v
Consumers (MUST be idempotent); outbox rows pruned afterwards
```

## Trade-offs

**Gains**

| Benefit | Why |
|---------|-----|
| Atomicity without 2PC | State + event in one local transaction — no distributed commit. |
| Reliable at-least-once | Event is durable in DB/WAL before any publish; a crash can't lose it. |
| Producer decoupled from broker | Write path doesn't call Kafka; DB writes succeed even if Kafka is down. |
| Per-aggregate ordering | WAL preserves commit order; key by `aggregate_id`. |
| Single source of truth | Events derived from committed state — state & stream can't diverge. |
| No DB polling load (vs polling relay) | Log tailing reads the WAL, not the table — lower latency, no missed rows. |

**Costs**

| Trade-off | Why it happens |
|-----------|----------------|
| **At-least-once, not exactly-once → duplicates** | Offset commits *after* publish; crash in between re-sends. **Consumers must be idempotent** (dedupe on event id). The most important consequence. |
| **Eventual consistency** | Relay is async; consumers see events after commit + pipeline lag, not synchronously. |
| **Postgres replication-slot risk** ⚠️ | A logical slot retains WAL until consumed. Debezium down/lagging → **WAL fills disk → primary DB down**. Alert on slot lag. |
| **Operational complexity** | You run Kafka Connect + Debezium: connectors, slots, schema-history topic, DLQ, offsets — a distributed system to operate. |
| **Failover fragility** | Logical slots historically don't survive primary→standby failover (pre-PG16); failover can force re-snapshot. |
| **Initial snapshot cost** | New connector snapshots existing tables first — heavy/locky for large tables (incremental snapshots mitigate). |
| **Write amplification** | Extra outbox row per transaction → more WAL + replication traffic. |
| **Outbox growth / bloat** | Table grows fast; needs pruning/partitioning + vacuum. (Debezium needs only the WAL, so rows can be deleted post-capture.) |
| **Schema/contract coupling** | Direct CDC leaks internal schema (refactor breaks consumers); outbox + Event Router + schema registry decouples but is more to maintain. |
| **Latency floor** | commit → WAL flush → Debezium → Connect → Kafka. Usually sub-second, non-zero, grows under lag. |
| **Poison messages** | At-least-once + malformed payloads → need DLQ + replay. |
| **No global ordering** | Kafka orders per-partition only; cross-aggregate order lost (fine if you only need per-aggregate). |
| **Harder to trace** | Event journey spans DB → WAL → Debezium → Kafka → consumer; needs correlation ids. |

## When to use / when not

- **Use it** when a service must reliably emit events on state changes at real
  volume, can't tolerate lost/orphan events, and eventual consistency is OK (the
  common event-driven-microservice case — e.g. emitting case/audit events to Kafka
  in the LoanApproval / AutonomousCustomerSupportAgent designs).
- **Prefer the polling relay** (no Debezium) at low volume or to avoid operating
  Kafka Connect — same atomicity, less ops.
- **Don't use it** when a reader needs the event *synchronously/transactionally*
  with the write (that's a local read, not an event), or when you think you need
  exactly-once across DB+broker (solve with idempotent consumers instead; full XA
  is almost never worth it).

## Related patterns

- **Idempotent consumer** — the mandatory companion (handles the at-least-once
  duplicates).
- **CDC / listen-to-yourself** — Debezium straight off business tables; same
  dual-write fix, but CRUD deltas instead of domain events.
- **Polling publisher** — the non-CDC relay.
- **Saga** — outbox is how each saga step reliably emits its next command/event.
- **2PC / XA** — the heavyweight alternative Outbox is designed to avoid.
