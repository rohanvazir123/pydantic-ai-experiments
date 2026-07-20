# Real-Time Aircraft Telemetry Processing — Design Doc

A Flink pipeline that ingests high-frequency eVTOL telemetry, restores order,
detects anomalies via rules + ML, and — the gap this doc closes — **enriches
each event with slow-changing context** (aircraft metadata, weather, flight
plan, maintenance history) before those detectors ever see it. Dedup and
enrichment are now built as a demo pipeline, kept in separate files from
`events.py` (see [Implementation: split files](#implementation-split-files));
rules engine and ML inference are still just the skeleton comment.

## Table of Contents

- [Where this fits](#where-this-fits)
- [Current state](#current-state)
- [Pipeline overview](#pipeline-overview)
- [Real-Time Metadata Enrichment](#real-time-metadata-enrichment)
  - [Why one join pattern doesn't fit all four sources](#why-one-join-pattern-doesnt-fit-all-four-sources)
  - [Aircraft metadata — broadcast state](#aircraft-metadata--broadcast-state)
  - [Maintenance history — broadcast state](#maintenance-history--broadcast-state)
  - [Weather — broadcast-refresh](#weather--broadcast-refresh)
  - [Flight plan — temporal (versioned) join](#flight-plan--temporal-versioned-join)
  - [Join pattern comparison](#join-pattern-comparison)
- [Where enrichment sits in the DAG](#where-enrichment-sits-in-the-dag)
- [Implementation: split files](#implementation-split-files)
- [Open questions / named limits](#open-questions--named-limits)
- [Proposed file layout](#proposed-file-layout)
- [Status](#status)

## Where this fits

`basics/telemetry/README.md` asks three questions of every station in this
system: **who owns this state, what bounds it, what advances time.** Every
exercise so far answers them with one shape — hash-partition by vehicle id,
one owner, no lock. Metadata enrichment is the first station in this repo
where that shape genuinely doesn't apply uniformly: the four side-inputs
named in the skeleton comment (aircraft metadata, weather, flight plan,
maintenance history) have four different sizes, update frequencies, and
keys, so "how do I join this?" has four different correct answers, not one.

That's also why this doc exists before code does — same reason
`job_sched/README.md` exists before `job_sched/scheduler.py`: pick the
pattern per side-input first, because retrofitting a broadcast join into
code written for a lookup join (or vice versa) means rewriting the operator,
not tuning a parameter.

## Current state

| Stage (skeleton) | Built? | Where |
|---|---|---|
| Source: MQTT → Kafka | No — `EvtolSensorSource` fabricates random readings | `events.py:97` |
| Deserialize → assign timestamps → watermarks | Yes (5s bounded-out-of-orderness, 500ms interval) | `events.py:159-167` |
| KeyBy(vehicle_id) | Yes (`AircraftIdSelector`, keyed by `aircraft_id`) | `events.py:170-173` |
| Deduplication | Yes — demo, per-key TTL'd last-seen check | `dedup.py` |
| **Real-Time Metadata Enrichment** | Yes — demo, broadcast + temporal join | `enrichment.py` |
| Sliding window | Yes (10 min window / 1 min slide) | `events.py:176-178` |
| Rules engine / CEP | No — only an inline `avg_temp > 55.0` threshold inside the window function, not a standalone rules stage | `events.py:132` |
| ML inference | No | — |
| Sink: Kafka, TSDB | No — `.print()` to console | `events.py:186` |

"Built" here means demo-level and syntax-checked, not runtime-verified —
PyFlink isn't installed in the environment this was written in, so the
pipeline hasn't actually been executed end to end. See
[Implementation: split files](#implementation-split-files) for how the new
stages compose with `events.py`.

The skeleton and scratch pseudocode at the top of `events.py` (lines 11–91)
describe the target shape; this doc formalizes one piece of it.

## Pipeline overview

```
MQTT ──► Kafka ──► Deserialize ──► Watermarks (5s, 500ms tick) ──► KeyBy(aircraft_id)
                                                                        │
                                                                        ▼
                                                                  Dedup (per key)
                                                                        │
                                                                        ▼
                                                        ┌─── Real-Time Metadata Enrichment ───┐
                                                        │  aircraft metadata · weather ·       │
                                                        │  flight plan · maintenance history   │
                                                        └───────────────┬───────────────────────┘
                                                                        ▼
                                                        Sliding Window ──► Rules Engine / CEP
                                                                        │
                                                                        ▼
                                                              ML Inference (feature
                                                              extraction → model →
                                                              anomaly score → alert)
                                                                        │
                                                                        ▼
                                                        Sink: Kafka (alerts) + TimescaleDB (raw + aggregates)
                                                                        │
                                                                        ▼
                                                    Dashboards / predictive maintenance / retraining feedback
```

Enrichment sits **after dedup, before windowing** — rules and ML both need
enriched context (e.g. "is this vibration high *given the current flight
phase*", "is this battery temp high *given ambient temperature*"), so
features must exist before the window aggregates or the rules engine
evaluates them. Enriching post-window would mean re-deriving per-event
context from an already-aggregated value, which is lossy and pointless.

## Real-Time Metadata Enrichment

### Why one join pattern doesn't fit all four sources

| Source | Size | Update frequency | Join key | Staleness tolerance |
|---|---|---|---|---|
| Aircraft metadata | Tiny (fleet size — dozens to low hundreds of rows) | Rare (fleet changes, recalibration) | `aircraft_id` | High — a stale airframe spec is still almost always correct |
| Maintenance history | Small (one record set per aircraft) | Event-driven (a service event happened) | `aircraft_id` | Medium — a missed update means a just-serviced aircraft is briefly judged against old history |
| Weather | Large, external, unbounded (every grid cell, continuously) | Frequent (minutes) | `(lat, lon, time)` grid cell — not `aircraft_id` | Low-medium — a few minutes stale is fine, hours is not |
| Flight plan | Small per flight, but versioned (amendments happen mid-flight) | Per-flight, occasional amendments | `aircraft_id` **and** time range (which version was active when) | None — must reflect the plan version active *at event time*, not processing time |

Size and key shape are what force the split: everything keyed by
`aircraft_id` and small enough to fit in memory on every parallel task can be
**pushed to every operator instance**; flight plan can be broadcast-sized but
not broadcast-*joined*, because "which version" is a function of event time,
not just key. Weather was originally going to be the exception that needed a
genuine per-event network call — but see the next section: that plan hit a
real gap in PyFlink's Python API, and weather ended up in the broadcast lane
too, just keyed by grid cell instead of `aircraft_id` and bounded to the
cells actually in play rather than the whole globe.

### Aircraft metadata — broadcast state

Flink's **Broadcast State Pattern**: a low-throughput `aircraft_metadata`
stream feeds a `BroadcastProcessFunction`/`KeyedBroadcastProcessFunction`
holding a `MapState<aircraft_id, AircraftMetadata>` replicated to every
parallel instance. The keyed telemetry stream reads that map on every
element — no network call, no per-event latency cost.

- **Ownership:** genuinely shared read-only data, replicated rather than
  partitioned — the one case in this repo's vocabulary where *copying* beats
  *owning*, because the data is small enough that the copy is cheaper than
  coordinating access to a single copy.
- **Bound:** fleet size. Explicitly named because broadcast state must fit
  in memory on **every** task manager, not just one place — unlike the
  hash-partitioned state everywhere else in this repo, this bound doesn't
  shrink by adding parallelism.
- **Time:** updates arrive on the broadcast stream itself and apply
  immediately; no polling, no sweep — same "push updates the state directly"
  shape as the rate limiter's token refill, just without the lazy-recompute
  trick, because there's no decay function to replay.

### Maintenance history — broadcast state

Same pattern and same reasoning as aircraft metadata — small, keyed by
`aircraft_id`, low update frequency. Kept as a **separate** broadcast state
(its own `MapStateDescriptor`) rather than merged into the aircraft-metadata
map, so a maintenance-system outage or schema change doesn't require
touching the aircraft-metadata path, and so each can be sourced from its own
upstream system (fleet database vs. maintenance system) without one feed
blocking the other.

### Weather — broadcast-refresh

The original plan here was Flink's **Async I/O API**
(`AsyncFunction` + `AsyncDataStream.unorderedWait`) — a genuine per-event
network call, capacity- and timeout-bounded, the same reasoning
`rag/v2/knowledge/bus/`'s circuit breaker already encodes for this repo: an
external dependency in a hot path must never block the pipeline thread. That
plan doesn't survive contact with the implementation: **Async I/O is a
Java/Scala-only operator — PyFlink's Python DataStream API doesn't expose
it.** There's no Python-side workaround that preserves the non-blocking
guarantee; a blocking call inside a `ProcessFunction` would stall the
pipeline thread on every cache miss, which is exactly what Async I/O exists
to prevent.

So weather moves into the same **broadcast state** mechanism as aircraft
metadata and maintenance history, with two differences: it's keyed by grid
cell instead of `aircraft_id`, and the broadcast stream is a periodic
*refresh* feed (a source polling the weather provider on an interval) rather
than an event-driven one (a service event or a fleet change). Weather
changes slowly enough — minutes, not seconds — that "replicate the latest
reading per cell, refreshed periodically" is a legitimate fit for the
problem, not just a workaround for the missing API.

- **Ownership:** same as aircraft metadata — replicated read-only data, not
  partitioned.
- **Bound:** grid cells actually in play (aircraft currently in flight), not
  the whole globe — the demo's `_synthetic_grid_cell` stand-in (see
  `enrichment.py`) makes this explicit by deriving a small, fixed set of
  cells rather than a real unbounded grid.
- **Time:** the refresh source's polling interval **is** the tick — the
  same "push updates the state directly" shape as aircraft metadata, just
  on a timer instead of on a real-world event.
- **What's now unsolved instead:** per-event freshness. Broadcast-refresh
  gives every event the *most recently polled* reading, not a reading fetched
  *for that event* — acceptable given weather's staleness tolerance (see the
  table above), but worth naming: if per-request freshness ever matters more
  than periodic refresh allows, that pushes the async-call requirement onto
  the Java/Scala side of the job (a Java `AsyncFunction` invoked from the
  Python API via a JAR, or a Table API async lookup join), not onto pure
  PyFlink.

### Flight plan — temporal (versioned) join

Flight plan is small and keyed by `aircraft_id` like the broadcast sources,
but broadcasting it wrong would silently give every event the *latest*
plan version rather than the version active *when the event happened* — the
in-flight amendment case. This needs an event-time-correct answer, which
broadcast state doesn't give: a plain `MapState<aircraft_id, FlightPlan>`
overwritten on every amendment has no memory of what was true a minute ago.

Flink's answer is a **temporal table join** — either the Table API's
`FOR SYSTEM_TIME AS OF <event_time>` versioned join, or the DataStream
equivalent: keyed state holding a **sorted list of `(valid_from, plan)`
versions per aircraft**, with the process function picking the latest
version whose `valid_from <= event_time` instead of just "the current one."

- **Ownership:** per-aircraft, so this partitions the same way everything
  else in this repo does — `keyBy(aircraft_id)`, one versioned-state entry
  per key, no cross-key coordination.
- **Bound:** versions per aircraft must be capped or GC'd (a flight plan
  amended 50 times shouldn't grow its state forever) — the same
  "state grows without a sweep" bug the parent README already names for
  `_status`/`_cancelled`; the sweep here is "drop plan versions older than
  the current flight."
- **Time:** this is the one enrichment source where correctness is
  event-time-sensitive, not just latency-sensitive — reusing the pipeline's
  own watermarks to pick the right version is what the other three sources
  don't need and this one can't skip.

### Join pattern comparison

| | Aircraft metadata | Maintenance history | Weather | Flight plan |
|---|---|---|---|---|
| Flink mechanism | Broadcast state | Broadcast state | Broadcast state (refresh feed) | Temporal (versioned) join |
| Join key | `aircraft_id` | `aircraft_id` | grid cell | `aircraft_id` + event time |
| Update trigger | Event-driven (fleet change) | Event-driven (service event) | Periodic poll | Event-driven (plan amendment) |
| Per-event cost | In-memory map read | In-memory map read | In-memory map read | In-memory versioned lookup |
| Correctness model | Eventually consistent (fine — low staleness sensitivity) | Eventually consistent | Eventually consistent, bounded by poll interval | Must be exact as-of event time |
| Failure mode | N/A (no external call) | N/A | Stale-until-next-poll if the provider is unreachable | N/A (all in local state) |
| Prior art in this repo | — (no exact precedent; closest is the rate limiter's "push updates the state directly") | Same as aircraft metadata | Same as aircraft metadata, on a timer | `resequencer.py`'s heap-of-versions shape, applied to plan versions instead of frame sequence numbers |

## Where enrichment sits in the DAG

All four joins attach to the **same keyed, watermarked stream**, after
dedup and before windowing — they don't need to run in a fixed order
relative to each other since they enrich disjoint fields, but they must all
complete before the sliding window or rules engine reads the event, since
both consume the enriched fields (ambient weather for battery-temp context,
flight phase for vibration context, maintenance recency for risk scoring).

## Implementation: split files

`events.py` stays untouched as the reference implementation — it still runs
standalone exactly as before. The new stages live in their own files rather
than being added to it, matching how this repo already splits by concern
(`worker_queues/base.py` vs. the CPU/IO-specific files):

- **`dedup.py`** — `DedupFunction`, a `KeyedProcessFunction` that drops an
  event repeating the last timestamp seen for its aircraft_id, using
  `ValueState` with a TTL so the "last seen" marker doesn't grow forever.
- **`stores.py`** — fake clients standing in for the real backing stores
  (Postgres for aircraft/flight-plan, a config service for firmware/operator,
  DynamoDB/Cassandra-shaped for maintenance history, Redis-shaped for the
  weather cache). Raw telemetry carries `vehicle_id, timestamp, motor_temp,
  rpm, gps, ...`; everything an ML model additionally needs (battery
  chemistry, motor model, firmware version, aircraft model, manufacturing
  date, maintenance history, flight plan, operator) but Kafka never carries
  lives here. Deliberately excludes battery age (derived, not looked up —
  feature engineering's job) and a feature-store client (deferred, see
  Status).
- **`enrichment.py`** — the two join patterns from the sections above:
  `AircraftContextEnrichment` (a `KeyedBroadcastProcessFunction` handling
  aircraft metadata, maintenance history, weather, firmware, *and* operator —
  all five broadcast) and `FlightPlanEnrichment` (a `KeyedCoProcessFunction`
  doing the as-of-event-time versioned lookup). Also holds the simulated
  update sources (`SimulatedContextUpdateSource`,
  `SimulatedFlightPlanUpdateSource`) that fabricate side-input data by calling
  into `stores.py`, the same role `EvtolSensorSource` plays for telemetry in
  `events.py`.
- **`pipeline.py`** — the orchestrator. Imports the reusable pieces of
  `events.py` (`EvtolSensorSource`, `EvtolTimestampAssigner`,
  `AircraftIdSelector`, `WindowAnomalyEvaluator`) rather than duplicating
  them, and wires: source → watermarks → keyBy → **dedup** →
  **broadcast enrichment** → **flight-plan temporal join** → window → sink.
  This is the file to run for the enriched pipeline; `events.py`'s own
  `run_evtol_pipeline()` is unaffected and still demonstrates the
  windowed-average stage on its own. Carries inline `PROD GOTCHA` comments
  at each spot that would need more work for a real deployment (no
  checkpointing, broadcast-input ordering isn't guaranteed, parallelism
  doesn't reflect a real partitioned topology) rather than building that
  work out — see those comments and the ones in `stores.py`/`enrichment.py`
  for the fuller failure-mode list.

None of this has been executed against a real Flink cluster or a local
`pyflink` install — see [Current state](#current-state) and
[Status](#status).

## Open questions / named limits

Named, not silently built around — matching this repo's habit
(`job_sched/README.md`'s "Open questions" is the direct precedent):

- **Weather grid resolution is undecided** — too coarse and enrichment is
  useless (whole-region weather for a fast-moving aircraft); too fine and
  broadcast state grows toward the same "doesn't fit in memory" problem the
  broadcast pattern is supposed to avoid.
- **Broadcast state has no eviction policy named here** — fine at current
  fleet size (dozens of aircraft), but "how large before broadcast stops
  being the right pattern" isn't answered.
- **Flight-plan version pruning policy is undecided** — when exactly old
  versions are dropped (on flight completion? a fixed retention window?)
  isn't picked yet.
- **What happens when maintenance-history and aircraft-metadata broadcast
  streams disagree on an `aircraft_id` neither has seen yet** (a race between
  first telemetry and first metadata for a newly added aircraft) is
  unresolved — likely "enrich with nulls until the broadcast catches up,"
  but not decided.

## Proposed file layout

- `anomaly/README.md` — this document
- `anomaly/events.py` — existing skeleton + windowed-average stage (built, untouched)
- `anomaly/dedup.py` — the dedup `KeyedProcessFunction` (built, demo-level)
- `anomaly/stores.py` — fake Postgres/config-service/DynamoDB-Cassandra/Redis clients (built, demo-level)
- `anomaly/enrichment.py` — the two join patterns and their simulated update sources (built, demo-level)
- `anomaly/pipeline.py` — orchestrator wiring dedup + enrichment into `events.py`'s stream (built, demo-level)
- `anomaly/rules.py` — standalone rules engine / CEP patterns, pulled out of the inline threshold in `WindowAnomalyEvaluator` (not built)
- `anomaly/ml_inference.py` — feature extraction + model scoring (not built)

## Status

**Enrichment: demo-built, not runtime-verified.** `dedup.py`, `stores.py`,
`enrichment.py`, and `pipeline.py` exist and are syntax-checked, but this was
written in an environment without a `pyflink` install, so the pipeline has
not actually been executed. Simulated side-inputs stand in for real MQTT/
Kafka/Postgres/config-service/DynamoDB/Redis/weather-API sources throughout —
see [Open questions](#open-questions--named-limits) for what's still
undecided, and the `PROD GOTCHA` comments in `pipeline.py`/`stores.py`/
`enrichment.py` for failure modes named but not solved (no checkpointing,
no retry/circuit-breaker on external calls, no broadcast-state eviction, no
flight-plan version pruning, no schema-evolution story).

**Battery age and feature-store lookups: explicitly deferred**, not
forgotten — age is a derived value computed from data `stores.py` already
provides, and feature-store point-in-time lookups belong to the
feature-engineering stage, not enrichment (see `stores.py`'s docstring).

**Rules engine, ML inference, real sinks: designed only, no code** — same
status as before this round of work; only the enrichment gap has closed.
