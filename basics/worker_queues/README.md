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
- [Running the Demos](#running-the-demos)
- [Running the Tests](#running-the-tests)

## Files

| File | Purpose |
|------|---------|
| `cpu_workloads.py` | Original CPU draft (kept for reference; do not build on it). |
| `cpu_workloads_fixed.py` | Reviewed, corrected `multiprocessing` worker pool. |
| `io_workloads.py` | Original IO draft (kept for reference; do not build on it). |
| `io_workloads_fixed.py` | Reviewed, corrected `asyncio` + `aiosqlite` worker pool. |
| `test_worker_queues_fixed.py` | Pytest suite for both fixed pools. |

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
   target is a `CpuWorker` instance whose only state is the (spawn-picklable)
   queues, an int id, and the processor — it never references the manager. The
   demo is guarded by `if __name__ == "__main__"`.

**`io_workloads.py` → `io_workloads_fixed.py`**

1. **Fake-async SQLite** — the original `await`ed the synchronous `sqlite3` API,
   which raises `TypeError`. Fixed: genuinely async access via `aiosqlite`.
2. **Wrong `task_done` target** — it called `queue.task_done()` on the imported
   `queue` *module* (and even `await`ed it). Fixed: `self.io_queue.task_done()`
   in a `finally`.
3. **`insert_io_task`** is now a coroutine using `await put`, so it applies
   backpressure at `maxsize` and can be awaited as callers expect.
4. Removed the bogus `from dbm import sqlite3` / unused `import io`, and moved
   the type check before attribute access.

## CPU-bound Pool

Split into three classes:

- **`ImageProcessor`** — the CPU-bound work itself (stateless, unit-testable;
  subclass or inject to do real work).
- **`CpuWorker`** — one worker process's consume loop; holds only
  spawn-picklable state so its `run` method is a valid `Process` target.
- **`CpuWorkerPool`** — owns the task/result queues and the pool of processes.

`CpuWorkerPool` starts one process per core (overridable), each running a
`CpuWorker` that pulls `ImageProcessingRequest` payloads from a `JoinableQueue`,
processes them via the injected `ImageProcessor`, and pushes results to a result
queue. Lifecycle:

```python
pool = CpuWorkerPool(num_workers=4)
pool.insert_cpu_tasks(payloads)
pool.join_tasks()                       # wait for all work to finish
results = pool.collect_results(len(payloads))
pool.shutdown()                         # sentinels first, then join
```

## IO-bound Pool

Same shape as the CPU pool, plus a connection pool:

- **`TelemetryData`** — the message model.
- **`SqliteConnectionPool`** — a bounded, coroutine-safe pool of reusable
  `aiosqlite` connections (opened once, borrowed per write).
- **`TelemetryWriter`** — the async persistence work; holds no connection of
  its own, just borrows one from the pool per `write()`.
- **`IoWorker`** — one asyncio worker's consume loop (creates its own writer
  over the *shared* pool).
- **`IoWorkerPool`** — owns the queue, the connection pool, and the workers.

`IoWorkerPool` starts N `asyncio` tasks, each an `IoWorker` that consumes
`TelemetryData` from an `asyncio.Queue` and persists each row by borrowing a
connection from the shared `SqliteConnectionPool`. Construction must happen
inside a running event loop.

### Connections, concurrency & SQLite nuances

- **Reuse, don't reconnect** — connections are opened once and reused across
  writes, instead of `connect()`-ing per row.
- **Coroutine-safe, not thread-safe — deliberately.** The pool is safe for
  concurrent *coroutines* on **one event loop** (which is exactly what this
  asyncio design needs), because the backing `asyncio.Queue`/`Lock` are
  loop-bound. It is **not** OS-thread-safe, and deliberately so: those
  primitives are loop-bound, and a SQLite connection generally can't be used
  from a thread other than the one that created it. If you truly need
  cross-thread sharing, the right pattern is a **pool (or loop) per thread**,
  not one pool shared across threads.
- **Pooling doesn't speed up writes.** SQLite serializes *all* writers behind a
  single database-level lock, no matter how many connections you open. The
  pool's real wins are: avoiding reconnect overhead, concurrent **reads**, and
  capping open handles.
- **Graceful degradation under contention.** Each pooled connection enables
  **WAL** (readers don't block the single writer) and a **`busy_timeout`**
  (wait-and-retry instead of an immediate "database is locked" error).
- **Tested** — `test_pool_handles_concurrent_writes` drives 15 overlapping
  writes through a 3-connection pool and asserts they all persist.

```python
q = IoWorkerPool(num_workers=5, db_path="telemetry.db")
await q.insert_io_task(TelemetryData(device_id="d1", metric={"temp": 21}))
await q.shutdown()                      # drain, sentinel each worker, gather
```

## Running the Demos

```bash
python cpu_workloads_fixed.py          # spins up a process pool, processes 20 images
python io_workloads_fixed.py           # writes 100 telemetry rows to telemetry.db
```

## Running the Tests

From the repo root, using the project venv:

```bash
.venv/bin/python -m pytest basics/worker_queues/test_worker_queues_fixed.py -v
```

Dependencies: `pydantic`, `aiosqlite`, `pytest`, `pytest-asyncio`. The CPU tests
start real worker processes; the IO tests write to a temporary SQLite file.
