# Worker Queues: CPU-bound vs IO-bound

Two small worker-pool patterns showing the right concurrency model for each
kind of work:

- **CPU-bound** (image processing, parsing, aggregation) → `multiprocessing`
  to get around the GIL and use multiple cores.
- **IO-bound** (telemetry writes, webhooks, DB logs) → `asyncio` so a handful
  of workers can overlap thousands of waiting I/O operations on one thread.

## Table of Contents

- [Files](#files)
- [CPU-bound Pool](#cpu-bound-pool)
- [IO-bound Pool](#io-bound-pool)
- [Resequencing Pool](#resequencing-pool)
- [Running the Demos](#running-the-demos)
- [Running the Tests](#running-the-tests)

## Files

| File | Purpose |
|------|---------|
| `worker_pool_base.py` | Shared abstractions both pools inherit: `Job` (self-processing), `Worker`, `WorkerPool` (pydantic). |
| `cpu_workloads.py` | Original CPU draft (kept for reference; do not build on it). |
| `cpu_workloads_fixed.py` | Reviewed, corrected `multiprocessing` worker pool. |
| `io_workloads.py` | Original IO draft (kept for reference; do not build on it). |
| `io_workloads_fixed.py` | Reviewed `asyncio` pool with a **SQLAlchemy** (in-memory SQLite) sink. |
| `resequencer.py` | Out-of-order telemetry resequencing: same `asyncio` pool, sink swapped for a bounded per-device min-heap. |

Tests live in `basics/telemetry/tests/` (see [Running the Tests](#running-the-tests)).

### Scalable DB via SQLAlchemy

The IO pool persists through **SQLAlchemy Core** against `sqlite:///:memory:`.
SQLAlchemy is DB-agnostic and manages its own connection pool, so there's no
hand-rolled pool — and the "work against a scalable database" answer is a
one-liner: **swap the engine URL for `postgresql://...`** and the pool/writer
code is unchanged. Talk levers on top of that: pool sized to write concurrency,
batched / `COPY` inserts, partition/shard by device or time, Timescale for
telemetry. (In-memory SQLite specifics: `StaticPool` + `check_same_thread=False`
so the `asyncio.to_thread` worker threads share one DB, and a `threading.Lock`
serialises writes — both drop away under Postgres.)

### Shared abstractions (`worker_pool_base.py`)

Both pools have the same shape — a typed message on a queue, N workers consuming
it until a sentinel — so that shape lives in one place:

| Base | Kind | Subclasses | Why this kind |
|------|------|-----------|---------------|
| `Job` | pydantic `BaseModel` | `ImageProcessingRequest`, `TelemetryData`, `TelemetryFrame` | a job is **data** — validated, serialisable, picklable across the process boundary; a compute job also **processes itself** via `process()` |
| `Worker` | `ABC` | `CpuWorker`, `IoWorker`, `ResequencingWorker` | a worker is **behaviour** over non-serialisable runtime state (queues, connections) — not a model; `start()` creates its own task/process |
| `WorkerPool` | `ABC` | `CpuWorkerPool`, `IoWorkerPool`, `ResequencingWorkerPool` | owns the queue + N workers; `start()` is a **template method** sequencing the four build steps each pool fills in |

`Job` (the merged base — formerly `WorkItem` + `Job`) carries an auto-generated
`job_id` for tracking/cancellation plus an optional `type` label. A job
overrides `process(...)` to do its own work, so the worker branches on nothing
and holds no processor — it just calls `job.process(<resource>)` (Command
pattern; new job types = new subclasses). The **resource** passed in is the
infrastructure the worker owns but the job needs: the CPU worker passes its
**sink registry** (`job.process(self.sinks)` — the job routes its result to
`file` or `http`), the IO worker passes its **writer** (`job.process(self.writer)`
— the job persists itself to the DB). Pure-compute calls (`job.process()` with
no resource) still work and just return the result — handy for unit tests. The
one thing the ABCs can't hide: the CPU side is **synchronous** (runs in child
processes) and the IO side is **asynchronous** (coroutines on one loop), so
`run` / `shutdown` (and `process`) are sync in the CPU subclasses and `async` in
the IO subclasses — same contract, two colours.

### Pool lifecycle: construction is inert, `start()` runs

`__init__` only records config; `WorkerPool.start()` is a **template method**
holding the one ordered sequence every pool follows, with each step abstract
because the three share the shape but no mechanics:

| Step | `CpuWorkerPool` | `IoWorkerPool` / `ResequencingWorkerPool` |
|------|-----------------|--------------------------------------------|
| `_init_queue` | `JoinableQueue` | `asyncio.Queue(maxsize)` (backpressure) |
| `_init_shared_state` | `Manager` proxies (cross-process) | plain dict/set (one loop, one thread) |
| `_init_sinks` | `file` / `http` registry, picked per job | the `TelemetryWriter` / `Resequencer` — one sink |
| `_init_workers` | build N workers, `start()` each | same |

Setup stays out of the constructor because every step has a runtime side effect
— `_init_workers` spawns processes (CPU) and needs a running event loop (IO). A
constructor calling overridable methods would let a half-built pool escape if a
step raised, and it forces subclasses to set config *before* `super().__init__()`.
It also pairs `start()` with the explicit `shutdown()` instead of hiding one half
of the lifecycle.

The same rule holds one level down: **the pool never creates a task or process.**
`_init_workers` builds workers and calls `worker.start()`; each worker creates its
own handle (`self.task` / `self.process`) and the pool joins it at shutdown. The
handle can only be built from `self.run`, so it cannot exist before the worker
does. The alternative — the pool creating it — would keep `run()`
execution-agnostic, but that abstraction is unusable here: `IoWorker.run` is a
coroutine awaiting an `asyncio.Queue`, and `CpuWorker` is built around
spawn-picklable state, so neither is portable to the other execution model.

### Job lifecycle: `insert_job` / `cancel_job` / `get_job_status`

`WorkerPool` declares three more abstract methods so callers can manage
individual jobs by `job_id`, tracked through a `JobStatus` enum
(`QUEUED → RUNNING → DONE`, or `CANCELLED`, or `UNKNOWN`):

| Method | CPU (sync) | IO (async for `insert_job`) |
|--------|-----------|------------------------------|
| `insert_job(job)` | mark `QUEUED`, `put` on the queue | same, `await put` (backpressure) |
| `cancel_job(job_id) -> bool` | flag + mark `CANCELLED` if still `QUEUED` | same |
| `get_job_status(job_id) -> JobStatus` | read the status map | read the status map |

Two design points worth calling out:

- **Cancellation is lazy.** You can't pull a specific item out of a
  `multiprocessing`/`asyncio` queue, so `cancel_job` just *flags* the id; the
  worker checks the flag when it reaches the job and skips it, marking it
  `CANCELLED`. It only succeeds while the job is still `QUEUED` (returns `False`
  for unknown/running/done) — best-effort by nature. (Same lazy-deletion idea as
  a priority-queue tombstone.)
- **Where the status lives differs by model.** The IO pool keeps a plain
  `dict[str, JobStatus]` + `set[str]` — safe because every worker is a coroutine
  on one loop. The CPU pool's workers run in **separate processes**, so a plain
  dict wouldn't be visible to them; it uses a `multiprocessing.Manager()` whose
  proxy `dict`s are shared live across the parent and all workers (and are torn
  down in `shutdown()`). This is the same "declare state to match the concurrency
  model" theme as the pools themselves.

### Why the `_fixed` files exist

The original drafts each had logic bugs. The `_fixed` versions correct them
while leaving the originals unchanged for comparison.

**`cpu_workloads.py` → `cpu_workloads_fixed.py`**

1. **Deadlock** — the original joined the worker processes *before* sending the
   poison-pill sentinels, so the workers never exited and `join()` hung forever.
   Fixed: `shutdown()` sends one sentinel per worker, then joins.
2. **Crash on invalid payload** — it read `payload.image_id` before the
   `isinstance` check. Fixed: validate before touching attributes.
3. **Spawn safety** — `target=self.process_task` pickled the whole *manager*
   instance (incl. its list of `Process` objects, which can't be pickled) under
   the `spawn` start method (macOS/Windows default). Fixed: the `Process`
   target is a `CpuWorker` instance whose only state is spawn-picklable (the
   queues, the sinks, an int id, the manager proxies) — it never references the
   pool manager. The
   demo is guarded by `if __name__ == "__main__"`.

**`io_workloads.py` → `io_workloads_fixed.py`**

1. **Fake-async DB** — the original `await`ed the synchronous `sqlite3` API,
   which raises `TypeError`. Fixed: a **SQLAlchemy** sink; the blocking driver
   runs off the loop via `asyncio.to_thread`.
2. **Wrong `task_done` target** — it called `queue.task_done()` on the imported
   `queue` *module* (and even `await`ed it). Fixed: `self.io_queue.task_done()`
   in a `finally`.
3. **`insert_job`** is a coroutine using `await put`, so it applies backpressure
   at `maxsize` and can be awaited as callers expect.
4. Removed the bogus `from dbm import sqlite3` / unused `import io`, and moved
   the type check before attribute access.

## CPU-bound Pool

- **`ImageProcessingRequest`** — a self-processing CPU job: `process(sinks)`
  does the compute **and** routes its result into the sink it names.
- **`ResultSink`** — where a result goes, chosen *per job*: `FileWriter`
  (durable, one JSON line, Manager lock for cross-process appends) or
  `HttpResponder` (POST back to a callback URL; stubbed as an outbox list).
- **`CpuWorker`** — one worker process's consume loop; holds the sink registry
  and only spawn-picklable state, so its `run` method is a valid `Process` target.
- **`CpuWorkerPool`** — owns the task/result queues, the sinks, and the workers.

`CpuWorkerPool` starts one process per core (overridable), each running a
`CpuWorker` that pulls `ImageProcessingRequest` payloads from a `JoinableQueue`,
calls `job.process(self.sinks)`, and (optionally) pushes results to a result
queue. Lifecycle:

```python
pool = CpuWorkerPool(num_workers=4)
pool.start()                            # build queue, state, sinks, workers
for job in payloads:                    # each job carries result_sink="file"|"http"
    pool.insert_job(job)
pool.join_tasks()                       # wait for all work to finish
results = pool.collect_results(len(payloads))
stored = read_results(pool.result_path) # what the file sink persisted
pool.shutdown()                         # sentinels first, then join
```

### Result sinks + FastAPI

Where a result goes is a property of the **job** (`result_sink`), not the
worker — so one image persists to a file while another is sent back over HTTP,
with no branching in the worker (it just calls `job.process(self.sinks)`). That
makes the pool a drop-in **FastAPI service layer**: the lifecycle methods map
1:1 onto routes, and because every `Job` is a pydantic model it *is* the request
body.

```python
pool = CpuWorkerPool()
pool.start()                             # build on app startup

@app.post("/jobs")                       # ImageProcessingRequest is the body
def submit(job: ImageProcessingRequest) -> dict[str, str]:
    pool.insert_job(job)                 # returns immediately
    return {"job_id": job.job_id}

@app.get("/jobs/{job_id}")               # poll status
def status(job_id: str) -> dict[str, str]:
    return {"status": pool.get_job_status(job_id)}

@app.delete("/jobs/{job_id}")            # best-effort lazy cancel
def cancel(job_id: str) -> dict[str, bool]:
    return {"cancelled": pool.cancel_job(job_id)}
```

The result comes back either by the `http` sink POSTing to the caller's callback
URL (push), or the `file` sink persisting it for a later `GET` (poll). Same
`Job` → `Worker` → `WorkerPool` shape works under FastAPI, a CLI, or a consumer.

## IO-bound Pool

Same shape as the CPU pool, over a SQLAlchemy sink:

- **`TelemetryData`** — the message model; `process(writer)` persists itself
  (async, since the work is I/O) via the writer it's handed.
- **`TelemetryWriter`** — persists a row via SQLAlchemy, off the loop with
  `asyncio.to_thread`.
- **`IoWorker`** — one asyncio worker's consume loop; hands each job the writer
  and calls `job.process(writer)` — no SQL, no branching.
- **`IoWorkerPool`** — owns the `asyncio.Queue`, the engine/writer, and the
  workers, plus the job-status/cancel bookkeeping.

`IoWorkerPool` starts N `asyncio` tasks, each an `IoWorker` that consumes
`TelemetryData` from an `asyncio.Queue` and persists it. Construction is inert;
`start()` is what runs, and it must be called inside a running event loop.

```python
q = IoWorkerPool(num_workers=5)                       # in-memory SQLite by default
q.start()                                             # needs a running event loop
await q.insert_job(TelemetryData(device_id="d1", metric={"temp": 21}))
await q.shutdown()                                    # drain, sentinel each, gather
```

Swap the store by passing an engine: `IoWorkerPool(engine=make_engine("postgresql://..."))`.

### SQLite write nuance (worth naming)

The in-memory SQLite sink serialises writes (one `StaticPool` connection + a
`threading.Lock`; SQLite has a single writer lock regardless). So more workers
don't buy write throughput *here* — that's a property of the store, not the
pool. Point the same pool at **PostgreSQL** (row-level locking) and N workers
become genuine N-way write concurrency, bounded by the connection pool. Rule of
thumb: size workers to the store's real write concurrency, and never run more
than the connection pool allows.

## Resequencing Pool

`resequencer.py` answers the other half of the worker-pool question: *ensuring
out-of-order data is resequenced correctly*. The key insight is that ordering
breaks on the **completion** side, not arrival — with N concurrent workers and
variable per-frame latency, worker A can start seq=5 before worker B finishes
seq=4 and still finish first. So frames arrive in order and still come out
scrambled.

Same shape as the IO pool, over the same `worker_pool_base.py` bases — `TelemetryFrame(Job)`
→ `ResequencingWorker(Worker)` → `ResequencingWorkerPool(WorkerPool)`. The only
swap is the sink: **`Resequencer` sits exactly where `TelemetryWriter` sits**,
the infrastructure the worker owns and hands to `frame.process(...)`. Per device
it keeps only an `expected` counter, a min-heap of early arrivals, and the time
its gap stalled.

Three details worth naming out loud:

- **Two bounds, not one.** A gap is presumed lost once *either* `max_buffer`
  (frames withheld) or `max_delay` (time withheld) trips — then the heap drains
  in seq order, stepping over each gap and recording it. They fail in opposite
  directions: a size bound alone leaves ordering *latency* unbounded (a device
  at 1 Hz sits on a gap for a minute before filling a 60-frame buffer), a time
  bound alone leaves *memory* unbounded. `close()` covers end-of-stream, where
  neither bound can fire because both are only tested from `submit()`.
- **No dedupe index.** Duplicates are caught by `_emit_ready` itself (`seq <
  expected` → drop), so no set mirrors the heap. Heap entries are
  `(seq, tiebreak, frame)`: without the `itertools.count()` tiebreak, a
  duplicate seq would compare the pydantic models and raise `TypeError`.
- **Shard, don't lock.** One `Resequencer` serialises every device through one
  event loop, but nothing couples one device to another — all state is keyed by
  device, so hashing `device_id` to a resequencer/process/Kafka partition scales
  it out. That's the partition-key model: ordering holds only *within* a
  partition, which is all per-device ordering needs.

The header comment carries the full step-by-step plus the known limits; the
inline comments reference those step numbers.

## Running the Demos

```bash
python cpu_workloads_fixed.py          # process pool, processes 20 images
python io_workloads_fixed.py           # asyncio pool, writes 10 rows via SQLAlchemy
python resequencer.py                  # 30 frames, seq=15 lost -> one recorded gap
```

## Running the Tests

All telemetry tests live in `basics/telemetry/tests/`. From the repo root:

```bash
.venv/bin/python -m pytest basics/telemetry/tests/ -v
```

Dependencies: `pydantic`, `sqlalchemy`, `fakeredis`, `pytest`, `pytest-asyncio`.
The CPU tests start real worker processes; the IO tests use an in-memory SQLite
DB (nothing written to disk).
