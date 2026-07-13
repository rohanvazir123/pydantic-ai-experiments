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
3. **Spawn safety** — `target=self.process_task` pickles the whole instance
   (incl. the queue) under the `spawn` start method (macOS/Windows default).
   Fixed: the worker loop is a module-level function, queues are passed as
   arguments, and the demo is guarded by `if __name__ == "__main__"`.

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

`CpuWorkerQueue` starts one process per core (overridable), each running a
module-level consumer loop that pulls `ImageProcessingRequest` payloads from a
`JoinableQueue`, processes them, and pushes results to a result queue. Lifecycle:

```python
pool = CpuWorkerQueue(num_workers=4)
pool.insert_cpu_tasks(payloads)
pool.join_tasks()                       # wait for all work to finish
results = pool.collect_results(len(payloads))
pool.shutdown()                         # sentinels first, then join
```

## IO-bound Pool

`IoWorkerQueue` starts N `asyncio` tasks that consume `TelemetryData` from an
`asyncio.Queue` and persist each row with `aiosqlite`. Construction must happen
inside a running event loop.

```python
q = IoWorkerQueue(num_workers=5, db_path="telemetry.db")
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
