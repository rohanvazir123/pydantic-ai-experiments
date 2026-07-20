# Priority Scheduler — Design Doc

A job scheduler that runs tasks by **safety priority level**, gated by an
**eligible-at** timestamp and ordered within a level by **deadline**
(earliest-deadline-first). This document evaluates three designs and their
trade-offs; nothing here is implemented yet.

## Table of Contents

- [Where this fits](#where-this-fits)
- [Requirements](#requirements)
- [Priority vocabulary (proposed)](#priority-vocabulary-proposed)
- [Ordering key](#ordering-key)
- [Approach A — In-process two-heap scheduler](#approach-a--in-process-two-heap-scheduler)
- [Approach B — Redis-backed distributed queue](#approach-b--redis-backed-distributed-queue)
- [Approach C — Postgres-backed durable job store](#approach-c--postgres-backed-durable-job-store)
- [Comparison](#comparison)
- [Recommendation](#recommendation)
- [Open questions / named limits](#open-questions--named-limits)
- [Proposed file layout](#proposed-file-layout)
- [Status](#status)

## Where this fits

`basics/telemetry/README.md` asks three questions of every station in this
system. The answer to the first one — **who owns this state?** — puts this
exercise in a different bucket than most of its siblings.

The resequencer and worker pool get to **partition by key** (`hash(device_id)
% N`) and make ownership local: a device's ordering only ever depends on that
device's own frames, so each partition can own a private heap with no lock.
A safety-priority scheduler cannot do this. Priority is a **global** total
order — a `DISTRESS` task in partition A must preempt a `ROUTINE` task in
partition B, so splitting the ready set by key breaks the one guarantee the
scheduler exists to provide. That makes this exercise a cousin of
`rate_limiter/`, not of `resequencer.py`: state that is genuinely shared, and
must pay for coordination rather than dissolve the problem via ownership.

The ordering *algorithm*, though, is exactly the resequencer's heap with the
key swapped, as the parent README already predicts — that part carries over
unchanged across every approach below; only where the shared state lives
changes.

## Requirements

- Every task has a **safety priority level** (small, fixed set of levels).
- Every task has an **eligible-at** timestamp — must not run before it.
- Every task has a **deadline** — among eligible tasks at the same priority,
  earliest deadline runs first.
- Must work as a **shared, possibly-distributed** service, not just a single
  in-process demo: multiple producers submitting tasks, multiple workers
  able to pull the next runnable one.

## Priority vocabulary (proposed)

```python
class SafetyPriority(IntEnum):
    DISTRESS = 0   # immediate risk to life/safety — preempts everything
    URGENT   = 1   # safety-relevant, time-critical, not yet distress
    SAFETY   = 2   # safety-relevant advisory, not urgent
    ROUTINE  = 3   # normal operational task, no safety implication
```

Borrowed from the ICAO/ITU radiotelephony distress-priority convention
(MAYDAY / PAN-PAN / SÉCURITÉ / routine traffic) to fit this repo's
vehicle-telemetry domain — a strawman, confirm or rename before building.

`IntEnum`, not `StrEnum`, and deliberately: `JobStatus` in
`worker_queues/base.py` is a `StrEnum` because job status has no natural
order. Priority *is* an order — `IntEnum` with `0 = highest` gives a
correct `heapq` / `ORDER BY` comparison for free. (The repo also has a
`Severity(str, Enum)` in `workflows/incident_response/models.py` — a
different domain, and a style already inconsistent with `StrEnum`; worth
not repeating that inconsistency here.)

## Ordering key

`(priority, deadline, tiebreak, task)` — the resequencer's `(seq, tiebreak,
frame)` with the key swapped, for the same reason: pydantic `Job` models
define no ordering, so two entries with equal `(priority, deadline)` would
otherwise fall through to comparing the task payloads and raise `TypeError`.
`tiebreak = itertools.count()`, exactly as in `resequencer.py`.

## Approach A — In-process two-heap scheduler

Two `heapq` min-heaps, mirroring the resequencer's single-heap-per-device
shape but split by *what* they order:

- `_pending`: keyed by `(eligible_at, tiebreak, task)` — not yet runnable.
- `_ready`: keyed by `(priority, deadline, tiebreak, task)` — runnable now.

`_promote(now)` pops everything off `_pending` with `eligible_at <= now` and
pushes it onto `_ready`. Called lazily at the top of `submit()` and
`get_next()` — the rate limiter's "compute on read" trick, so idle time
costs nothing.

**Ownership:** one event loop, synchronous heap ops, no `await` inside
them — the same guarantee that lets the resequencer's `submit()` skip a lock.

**Liveness gap, named not fixed** (same shape as the resequencer's own
documented gap): if nothing calls `submit()`/`get_next()` after a task's
`eligible_at` passes, `_promote` never runs and that task waits until the
next call. Fine for a polling caller; a push-style consumer (workers parked
waiting for work) needs a periodic sweeper task
(`asyncio.sleep(tick)` → `_promote(now)`), exactly the fix the resequencer's
own README section names but doesn't build.

**Trade-offs**
- \+ Zero infra, microsecond ops, trivially unit-testable with a fake clock
  (the moving-average/resequencer testing pattern already in this repo).
- \+ This is the ordering *reference implementation* — every other approach
  should produce the same ordering as this one, given the same inputs.
- − Single process: a crash loses every pending and ready task. No
  durability.
- − Doesn't scale past one process's throughput ceiling, and can't be
  partitioned (see "Where this fits") to get around that.

## Approach B — Redis-backed distributed queue

Because the ready-state is genuinely shared (not partitionable by key), this
pays the rate limiter's kind of price: coordination, not ownership.

- **Delay set:** one ZSET, `sched:pending`, scored by `eligible_at`.
- **Ready sets:** one ZSET *per priority level* — `sched:ready:distress`,
  `...urgent`, `...safety`, `...routine` — scored by `deadline`. Kept
  separate rather than packed into one composite score, so each level stays
  independently inspectable (`ZRANGE ... WITHSCORES`) and there's no
  bit-packing arithmetic to get wrong.
- **Promotion (the sweeper):** periodically `ZRANGEBYSCORE sched:pending -inf
  <now>`, then move each match into its priority's ready set. This
  read-then-move is the same multi-key read-modify-write shape
  `rate_limiter/` already solves with `WATCH`/`MULTI`/`EXEC`. A Lua script
  (`EVAL`) is the alternative — one round trip, atomic server-side, no retry
  loop — and arguably simpler here since a periodic sweep has no per-caller
  contention to retry against, unlike the rate limiter's concurrent bucket
  writers.
- **Pop:** a worker tries `ZPOPMIN sched:ready:distress`, then `urgent`,
  `safety`, `routine` in order, falling through on empty. `ZPOPMIN` is
  atomic, so concurrent workers never double-pop. Using `task_id` as the
  ZSET member (not a serialized tuple) sidesteps the resequencer's tiebreak
  problem entirely — ZSET membership is by key, not by comparison, so two
  equal-deadline tasks never collide.
- **Durability is a config knob, not a guarantee:** RDB snapshotting loses
  everything since the last snapshot; AOF with `appendfsync=everysec` bounds
  loss to ~1s; no persistence loses everything on restart. `rate_limiter/`
  never had to face this — rate-limit state is disposable by nature;
  scheduled safety tasks are not.
- **At-least-once delivery is not free:** `ZPOPMIN` removes a task before
  its worker confirms execution, so a worker crashing between pop and
  completion silently loses it. This repo already has the right primitive
  for that problem elsewhere — `rag/v2/knowledge/bus/`'s Redis Streams
  consumer groups (`XPENDING`/`XCLAIM` for redelivery) — reuse that instead
  of inventing acks on top of ZSETs, if at-least-once matters.

**Trade-offs**
- \+ Multiple producers and workers, across hosts; survives a single
  scheduler process crashing.
- \+ Reuses two patterns already proven in this repo (optimistic-locking
  move, Streams redelivery) instead of inventing new ones.
- − Every op is a network round trip — latency floor is Redis RTT, not
  memory access.
- − Durability must be deliberately configured; this repo's default
  docker-compose Redis is unlikely to have AOF on.
- − Not linearly scalable by adding workers: the ready-set cardinality is
  fixed at "one per priority level," so Redis itself is the ceiling, not
  the partition count.

## Approach C — Postgres-backed durable job store

This repo already runs this exact pattern for a different job type —
`rag/v2/knowledge/scheduler/` (APScheduler + `job_store.py`, `croniter`,
migration `007_scheduler.sql`) — so this is prior art, not a hypothetical.

One table: `(task_id, priority, eligible_at, deadline, payload, status,
claimed_by, claimed_at)`.

```sql
SELECT * FROM tasks
WHERE status = 'pending' AND eligible_at <= now()
ORDER BY priority ASC, deadline ASC
LIMIT 1
FOR UPDATE SKIP LOCKED;
-- then: UPDATE ... SET status = 'claimed', claimed_by = $worker
```

`FOR UPDATE SKIP LOCKED` gives the same atomic-claim-without-blocking
property `ZPOPMIN` gives in Redis, using only a transaction — no separate
delay structure or promotion/sweep step needed at all, since the `WHERE`
clause *is* the eligibility check and is always correct as of query time.

**Trade-offs**
- \+ Full ACID durability for free; the Postgres instance already in this
  stack needs no new infra.
- \+ Priority + deadline ordering is one `ORDER BY`; eligibility is one
  `WHERE`. No promotion/sweep code to write or get wrong.
- \+ Auditable, queryable job history — a `GET /jobs/{id}` endpoint (the
  parent README's one deliberately-shared piece of state) is a plain read.
- − Polling-based unless paired with `LISTEN`/`NOTIFY`; a bare poll loop
  puts a latency floor at the poll interval, and `LISTEN`/`NOTIFY` removes
  that floor at the cost of another moving part.
- − Slowest per-op latency of the three (disk-backed transaction vs. Redis
  RTT vs. in-memory) — matters only if "safety priority" implies
  sub-millisecond dispatch.

## Comparison

| | A: In-process heap | B: Redis | C: Postgres |
|---|---|---|---|
| State ownership | single process | shared, coordinated | shared, transactional |
| Durability | none | config-dependent | full ACID |
| Pop latency | microseconds | network RTT | disk-backed transaction |
| Eligibility mechanism | delay heap + lazy/ticked promote | delay ZSET + sweep (WATCH/MULTI or Lua) | one `WHERE` clause, no sweep |
| At-least-once delivery | not solved | needs Streams (bus/ pattern) | solved (transaction) |
| Existing prior art here | `resequencer.py` | `rate_limiter/`, `bus/` | `rag/v2/knowledge/scheduler/` |
| Best for | ordering reference impl, tests | sub-10ms dispatch, ok with config'd durability | correctness/audit first, ok with ms-to-tens-of-ms latency |

## Recommendation

Layered, not a single winner in isolation — the ordering algorithm (the
enum, the comparison key, the promotion policy) is identical across all
three; only where the shared state lives changes, which is the parent
README's own lesson restated ("distribution happens at the queue").

1. **Build Approach A first**, as the ordering reference implementation and
   unit-test target — the same role `resequencer.py` already plays. Every
   other approach's output should match this one given the same inputs and
   a fake clock.
2. **For production, default to Approach C (Postgres)** if a poll- or
   `LISTEN`/`NOTIFY`-based latency floor in the tens-of-milliseconds is
   acceptable. It needs no new infrastructure, gives durability and audit
   history for free, and directly reuses the `scheduler/` module pattern
   already proven in this repo.
3. **Reach for Approach B (Redis) only if that floor is actually
   disqualifying** — i.e., true sub-10ms dispatch is a hard requirement.
   Accept that durability becomes a deliberate persistence-config decision,
   and reuse `rate_limiter/`'s optimistic-locking pattern (or a Lua script)
   for the pending→ready move, plus `bus/`'s Streams consumer groups if
   at-least-once delivery is also required.

## Open questions / named limits

Named, not silently built around — matching this repo's habit of writing
down what a design doesn't solve rather than hiding it:

- **Starvation:** strict priority ordering never lets `ROUTINE` run under
  sustained `DISTRESS`/`URGENT` load. No aging/anti-starvation policy is
  proposed here.
- **Missed-deadline policy is undecided:** when promotion finds a task
  whose `deadline` has already passed, this doc doesn't pick
  run-anyway vs. drop vs. escalate — needs a decision before implementation.
- **Priority vocabulary is a strawman** (`DISTRESS/URGENT/SAFETY/ROUTINE`) —
  confirm or replace the names/levels before building.
- **At-least-once delivery** is only solved by Approach C (transaction) and
  the Streams variant of Approach B — the plain `ZPOPMIN` version and
  Approach A both silently drop a task if its claimer dies mid-execution.

## Proposed file layout

Not created yet — for reference once an approach is approved:

- `job_sched/README.md` — this document
- `job_sched/scheduler.py` — Approach A, same shape as `resequencer.py`
- `job_sched/test_scheduler.py` — pytest + fake clock, added to
  `basics/telemetry/tests/conftest.py`'s `sys.path` list
- `job_sched/pg_scheduler.py` / `job_sched/redis_scheduler.py` — whichever
  of B/C gets picked for the production path, once decided

## Status

**Designed, not built** — all three approaches, matching the parent
README's own status table for this exact exercise.
