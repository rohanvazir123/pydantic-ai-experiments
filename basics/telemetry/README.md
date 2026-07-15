# Telemetry

A set of exercises that look like six unrelated puzzles — a rate limiter, a
worker pool, a producer–consumer queue, a moving average, a log parser, a
priority scheduler — and are actually **one system seen from six angles**:
high-frequency telemetry from many vehicles, ingested over queues, processed
concurrently, and served back without lying to anyone.

This file is the mental model that connects them. Each folder has its own README
for the algorithm and its bugs; this one is the spine.

## The mental model

Every station in a streaming system answers the same three questions. Once you
have them, each exercise stops being a puzzle and becomes an instance.

> **1. Who owns this state?**
> One owner per key means no lock — not a lock you chose well, *no lock at all*.
> Shared state is what forces a mutex, a proxy, or optimistic retries. So the
> first question is never "which lock?", it's "why is this shared?"
>
> **2. What bounds it?**
> A stream is infinite; your memory is not. Every buffer needs a bound — and
> usually **two**, because bounding memory and bounding latency are different
> problems and either one alone leaves the other unbounded.
>
> **3. What advances time?**
> If correctness depends on the passage of time, something must notice time
> passing. Either the state is **read** often enough to re-check lazily, or
> something must **tick** it. A buffer nobody reads and nothing ticks is a bug
> waiting for a quiet device.

Everything below is these three questions, worked out.

## Follow one frame

A vehicle emits a frame. Trace it to a stored answer and you pass every exercise
in the repo, in order:

```
  vehicles                                                  ┌── Resequencer[i]     order
     │                                                      │     (seq gaps, loss)
     ▼                                                      │
  ① gate ──► ② handoff ──► ③ router ──► ④ worker[i] ───────►├── RollingAverage[i]  windows
  (admit/drop)  (queue)     (hash        (the loop)         │     (last N seconds)
                             device_id)       │             │
                                              │             └── FlightState[i]     lifecycle
                                              │                                        │
                                              │                                        ▼
                                              │                                     ⑤ store
                                              │                                        │
                       _status / _cancelled ──┘                                        ▼
                       (the ONLY shared state)                                    ⑥ log parser
                                              ▲                                    (offline)
                                     gc worker ┘
```

| | Station | The exercise | Folder |
|---|---------|--------------|--------|
| ① | **Gate** — millions of frames arrive; decide who gets in | distributed token bucket | [`rate_limiter/`](rate_limiter/README.md) |
| ② | **Handoff** — the frame crosses from an ingress thread to a worker | producer–consumer, no deadlock, no races | [`worker_queues/`](worker_queues/README.md) |
| ③ | **Router** — which worker owns this vehicle? | worker pool | [`worker_queues/`](worker_queues/README.md) |
| ④ | **Work** — restore order, window, track lifecycle | resequencer · moving average · FSM | [`worker_queues/`](worker_queues/README.md) · [`moving_average/`](moving_average/README.md) · [`state_machine/`](state_machine/README.md) |
| ⑤ | **Store** — low-latency writes, fast reads | SQLAlchemy sink (SQLite → Postgres) | [`worker_queues/`](worker_queues/README.md) |
| ⑥ | **Exhaust** — find anomalies after the fact | log parser *(designed, not built)* | — |

The arc worth holding onto: **② is the atom and ③ is the molecule.** A worker
pool *is* a producer–consumer queue, plus routing, plus sinks, plus a lifecycle.
Get the two-line handoff wrong and nothing above it can be right. And a priority
scheduler *(also designed, not built)* is the resequencer's heap with the key
swapped — `(priority, deadline, task)` where the resequencer has
`(seq, tiebreak, frame)`, including the same tiebreak trick, because pydantic
models define no ordering and equal keys would otherwise compare the payloads and
raise `TypeError`.

## Question 1 — Who owns this state?

**The answer that makes everything else fall out: one owner per key.**

### Manager, not pool

A *pool* means interchangeable workers fed from one queue. Once each worker owns
its own queue and its own sinks, nothing is interchangeable, and the class owning
routing, global state, lifecycle, and non-job workers isn't a pool — it's a
**manager**.

### One queue per worker

The invariant is **worker ↔ queue, 1-1**. "One queue per vehicle" is a *routing
policy*, and it's the wrong one taken literally: it binds worker count to vehicle
count, so 10k vehicles means 10k workers — impossible for process-backed workers,
and a direct violation of the rule `worker_queues/` already states (size workers
to the store's real write concurrency; never exceed the connection pool).

Hash the key instead — `hash(vehicle_id) % N`. The same vehicle always lands in
the same queue, so per-vehicle ordering survives while N stays yours to choose.
The manager owns the router; workers stay dumb single-queue consumers.

| Policy | Ordering | Balance |
|--------|----------|---------|
| `hash(vehicle_id) % N` | per-vehicle, guaranteed | hot partitions (one chatty vehicle) |
| round-robin | none | even |

**What you give up is work-stealing.** A shared queue load-balances for free: any
idle worker takes any job. Under 1-1, an idle worker cannot help a saturated
neighbour. That is the cost, and it should be chosen, not discovered.

### What partitioning dissolves

`worker_queues/resequencer.py` exists because ordering breaks on the
**completion** side: N concurrent workers on a shared queue, variable latency, so
worker A finishes seq=5 before worker B finishes seq=4 and frames that *arrived*
in order come out scrambled.

Route by vehicle to one queue with one worker and there is no concurrency within
a vehicle. Completion order equals arrival order; the heap, the timeout, and the
gap heuristics have nothing to do.

It does **not** retire the Resequencer — arrival can still be genuinely
out-of-order or lossy (UDP; the never-delivered `seq=15` in the demo). But each
worker then keeps its own private buffer for the real network case. *Architecture
dissolving a problem beats an algorithm solving it.*

### Private sinks, and the death of every lock

**Every lock in these files exists only because something is shared.**

| Lock today | Why it exists | After partitioning |
|-----------|---------------|--------------------|
| `Resequencer` (none — relies on `submit()` being sync) | one heap, N workers | private heap; the lock question stops existing rather than being answered |
| `FileWriter`'s `Manager.Lock` | N processes appending to one file | one file per worker; nothing to serialise |
| `HttpResponder`'s shared outbox | one list, N processes | private; each worker POSTs independently |
| `TelemetryWriter`'s `threading.Lock` | SQLite `StaticPool` hands every caller the *same single connection* | it's the store, not the sink — see below |

But "private sink" cashes out differently depending on whether the sink **holds
state**:

- **`Resequencer` holds real state** (heaps, `_expected`, `_blocked_since`).
  Private is *essential* — it is the whole ordering-correctness argument, and it
  is what would let `submit()` become async (a real DB sink) later.
- **`TelemetryWriter` holds none.** It is an engine reference plus a lock; the
  state lives in the DB. A private writer is cosmetic, and the `Engine` must
  **stay manager-owned** — a SQLAlchemy engine *is* a connection pool, thread-safe
  and built to be shared. Give each worker its own and you get N separate
  in-memory databases. What kills that lock is the store: point it at Postgres and
  each worker checks out its own connection.
- **`FileWriter` is where ownership genuinely pays** — a private file per worker
  removes the `Manager` lock outright.

**The cost is real: private sinks fragment the output.** `resequencer.emitted` is
one dict today; with N private resequencers, reading results becomes a
scatter-gather, and per-worker files need a merge. A **DB sink escapes this** —
rows land in one table no matter which worker wrote them. An underrated argument
for a real store over in-memory sinks once you partition.

### Composition, not identity

A worker should **own** a private `Resequencer`, not **be** one. Merging them
kills the property the workers are built on ("branches on nothing, knows nothing
about heaps or sequence numbers"), costs the standalone algorithm tests, and makes
the worker the smartest object in the system. Same for the moving average and the
FSM: per-worker instances the worker owns and pokes.

### What stays shared

Exactly one thing: **`_status` / `_cancelled`** — because `GET /jobs/{id}` must
answer for any job without knowing which worker owns it.

That's a good property. The distribution seam narrows to a single map: the only
thing needing `Manager` proxies today, the only thing that would become Redis, and
precisely what the GC worker reaps. Everything else travels with its worker.

### The exception that proves the rule

The rate limiter is the one place state is **genuinely** shared — millions of
requests across many ingress nodes, one bucket per client. There is no owner to
give it to, so it pays the full price: Redis, and `WATCH`/`MULTI`/`EXEC`
optimistic locking to make read-modify-write safe under concurrent writers.

That contrast is the lesson. Everywhere a key has a natural owner, ownership
removes the lock. Where it truly doesn't, you get distributed consensus machinery
— and you can see the cost sitting right there in the retry loop.

### Distribution: the queue is the seam

If workers move to another host or state moves to Redis, **distribution happens at
the queue — not at `Worker.start()`.**

A manager cannot `create_task` on another host. A remote worker is a separate
process constructing its own worker from config; what crosses the wire is the job
payload. So remote execution is a `_init_queue` change (a Redis stream instead of
an `asyncio.Queue`), and remote state is a `_init_shared_state` change. Both are
already abstract steps of the `start()` template, and the CPU pool already proves
the indirection works — `Manager` proxies are *already* IPC over a socket, not
real shared memory.

Hash-routed partitions, one consumer each, keyed by vehicle, over a durable log
**is** the Kafka / Redis-Streams consumer-group model.

## Question 2 — What bounds it?

**Two bounds, because memory and latency are different problems.**

The resequencer states it best: `max_buffer` alone leaves *latency* unbounded (a
1 Hz vehicle sits on a gap for a minute before filling a 60-frame buffer);
`max_delay` alone leaves *memory* unbounded (a fast vehicle withholds a huge heap
inside the window). You need both. This generalises — any buffer holding data back
for a reason needs a size bound *and* a time bound.

| Station | Memory bound | Latency / liveness bound |
|---------|--------------|--------------------------|
| Gate | bucket capacity (burst) | refill rate (sustained) |
| Handoff | queue `maxsize` → backpressure | — |
| Resequencer | `max_buffer` | `max_delay` |
| Moving average | the window itself — samples age out | window length |
| `_status` map | **none — this is a live leak** | — |

**Where the bound goes when you partition.** Backpressure becomes per-queue, so a
hot vehicle blocks only its own producer — better isolation, but head-of-line
blocking moves to **ingress**, and ingress wants a **drop** policy, not a wait.
That is exactly the gate's job: under 1-1 routing, the rate limiter stops being a
separate exercise and becomes the thing that keeps a hot partition from stalling
the world.

Shutdown gets simpler too: one sentinel per queue for its one worker, rather than
N sentinels on a shared queue and the subtle assumption that each worker takes
exactly one.

## Question 3 — What advances time?

**Pull re-checks for free. Push needs a tick.**

The moving average and the resequencer take opposite sides of the same problem,
and the contrast is the reason to read them together:

| | `TelemetryRollingAverage` | `Resequencer` |
|---|---|---|
| Output is | **pulled** (`get_moving_average()`) | **pushed** (into `emitted`) |
| Expiry trigger | every read evicts first | only `submit()` re-checks the bounds |
| Vehicle goes quiet | self-correcting — the next read is still right | withholds forever; only `close()` frees it |
| Needs a sweeper | no | **yes** |

A **pull**-based sink gets its tick for free: a read is a natural place to
re-check time. A **push**-based sink has no reader, so nothing re-checks, so
something external must poke it.

The moving average already fixed the exact bug the resequencer still has — its
draft only evicted inside `add_batch`, so a quiet metric reported a stale average;
the fix was to evict on every read. It kept the door open both ways, too:
`add_batch([])` is a valid empty poke that just refreshes the window.

**The rate limiter is the same family.** Tokens refill *lazily on each request*
from elapsed time, rather than a background job topping up buckets. Compute-on-read
beats a timer: no scheduler, no drift, and idle buckets cost nothing.

### The live consequence

`Resequencer` is push-based with no tick, so **`max_delay` never fires for a quiet
vehicle**. Both bounds are only ever read from `submit()`, so a timed-out heap
doesn't emit until that vehicle's *next* frame arrives — and if it went quiet
mid-gap, never. `max_delay` is therefore not a latency bound at all: effective
release is `max_delay` *plus* the inter-frame interval, and it fails exactly when
something has already gone wrong.

### The sweep must follow ownership

A central GC reaching into every worker's private buffer re-introduces the sharing
that partitioning just removed — and it's the first thing to break if workers move
off-host. So it splits:

- **Per-worker sweep** for per-worker state. It falls out beautifully:
  `asyncio.wait_for(self.queue.get(), timeout=tick)` makes the worker's own idle
  time the sweep trigger — and "this worker's queue went quiet" is precisely the
  failure case `max_delay` cannot see. The worker stays dumb: it pokes; the
  deadline policy stays in the sink.
- **Central GC** for what is genuinely global — reaping `_status` / `_cancelled`,
  which the manager owns and which grow without bound today (every `job_id` ever
  seen stays forever, and `collect_results()` returns the whole dict). Per-vehicle
  state is never reclaimed either, in the resequencer or in a per-vehicle average.

## Where each piece fits

**`moving_average/` — a sink, or a post-processor after one.**

- Its concurrency contract is already the one partitioning wants: *single-owner,
  no internal lock, synchronous methods a single loop serialises for free.*
  Identical to `Resequencer.submit()`. Under private-per-worker that stops being a
  caveat in a docstring and becomes structural.
- **It is order-insensitive, which tells you when you need the resequencer at
  all.** A mean is commutative and windowed by *timestamp*, not sequence — it sorts
  arrivals internally and doesn't care what order they land in. If your only
  downstream is an aggregate, **skip resequencing**: its buffer, latency, and gap
  heuristics are only worth paying for order-*sensitive* sinks. Chain them
  (`worker → Resequencer → RollingAverage`) only when something downstream needs
  the order.

**`state_machine/` — a per-vehicle sink that *is* order-sensitive.** Transitions
are only meaningful in sequence, so this is the case that justifies putting the
resequencer in front.

**`rate_limiter/` — the gate**, and the ownership counter-example (above).

**`worker_queues/basic_producer_consumer*.py` — the atom.** The safe handoff every
other station is built on.

## Status

**Designed, not built.** The code is one stage behind this document:

| | Built | Designed here |
|---|-------|---------------|
| Class | `WorkerPool` | `WorkerManager` |
| Queues | one shared, N workers fan out | 1-1 worker ↔ queue, hash-routed |
| Sinks | one shared instance | private per worker |
| Locks | `threading.Lock`, `Manager.Lock`, sync-by-convention | none — ownership removes the need |
| Sweeper / GC | none (the leaks above are live) | per-worker sweep + central GC for `_status` |
| Post-processors | `moving_average/`, `state_machine/` standalone, unwired | per-worker sinks |
| Gate | `rate_limiter/` standalone, unwired | admission control at ingress |
| Log parser · priority scheduler | — | ⑥ and the heap-with-a-different-key |

What *is* built and current: the `start()` lifecycle — construction is inert,
`WorkerPool.start()` is a template method sequencing four abstract steps, and each
worker creates and owns its own task/process. See
[`worker_queues/README.md`](worker_queues/README.md).

## Running things

```bash
# All telemetry tests (no services needed)
.venv/bin/python -m pytest basics/telemetry/tests/ -q

# Demos
cd basics/telemetry/worker_queues
python basic_producer_consumer_fixed.py   # the atom: safe handoff
python cpu_workloads_fixed.py             # 4 processes, file + http sinks
python io_workloads_fixed.py              # asyncio + SQLite via SQLAlchemy
python resequencer.py                     # workers scramble; the heap restores order

python ../moving_average/moving_average_fixed.py   # window ages out under a fake clock
python ../state_machine/flight_state_machine.py    # invalid transitions raise
```
