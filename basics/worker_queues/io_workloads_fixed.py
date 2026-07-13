# =====================================================================
# IO-BOUND worker pool (asyncio: telemetry APIs, webhooks, DB logs)
# =====================================================================
#
# Fixed/reviewed version of ``io_workloads.py``. The original is left unchanged
# for reference; this file corrects the logic bugs.
#
# Responsibilities are split into classes that mirror the CPU version:
#   * ``TelemetryData``          — the message model
#   * ``SqliteConnectionPool``   — a bounded, coroutine-safe pool of aiosqlite
#                                  connections, reused across writes
#   * ``TelemetryWriter``        — the actual (async) persistence work
#   * ``IoWorker``               — one asyncio worker's consume loop
#   * ``IoWorkerPool``          — owns the queue and the pool of worker tasks
#
# Bugs fixed vs the original
# --------------------------
# 1. Fake-async SQLite: the original did ``await sqlite3.connect(...)`` etc.
#    ``sqlite3`` is synchronous, so those awaits raise ``TypeError``. This
#    version uses ``aiosqlite`` for genuinely async database access.
# 2. Wrong ``task_done`` target: the original called ``queue.task_done()`` on
#    the imported ``queue`` *module* (and even ``await``ed it). Here it is
#    ``self.io_queue.task_done()``, called once per ``get()`` in a ``finally``.
# 3. ``insert_io_task`` is now a coroutine using ``await self.io_queue.put(...)``
#    so it applies backpressure at ``maxsize`` and can be awaited as callers expect.
# 4. Removed the bogus ``from dbm import sqlite3`` and unused ``import io``.
# 5. Type validation happens before any attribute access.
#
# Connection handling, concurrency & SQLite nuances
# --------------------------------------------------
# Rather than reconnecting on every write (wasteful) or giving each worker its
# own dedicated connection, connections are pooled in ``SqliteConnectionPool``
# and borrowed per write.
#
# Thread-safety scope (deliberate):
#   * The pool is safe for concurrent *coroutines* on ONE event loop — which is
#     exactly what this asyncio design needs — via an ``asyncio.Queue`` of
#     connections guarded by an ``asyncio.Lock``.
#   * It is NOT OS-thread-safe, and deliberately so: ``asyncio.Queue``/``Lock``
#     are loop-bound, and a SQLite connection generally cannot be used from a
#     thread other than the one that created it.
#   * If you truly need cross-thread sharing, the right pattern is a pool (or
#     event loop) PER THREAD — not one pool shared across threads.
#
# SQLite write caveat: pooling does NOT speed up writes — SQLite serializes all
# writers behind a single database-level write lock regardless of how many
# connections you open. The pool's real wins are:
#   * no per-write connect overhead (connections are reused),
#   * concurrent *reads*, and
#   * a fixed cap on open handles.
# To make write contention degrade gracefully, each pooled connection enables
# WAL mode (readers don't block the single writer) and a busy_timeout
# (wait-and-retry instead of an immediate "database is locked" error).
# ``test_pool_handles_concurrent_writes`` exercises 15 overlapping writes
# through a 3-connection pool to confirm they all persist safely.

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiosqlite
from pydantic import BaseModel

TELEMETRY_DDL = "CREATE TABLE IF NOT EXISTS telemetry (device_id TEXT, metric TEXT)"


class TelemetryData(BaseModel):
    device_id: str
    metric: dict


class SqliteConnectionPool:
    """A bounded, coroutine-safe pool of reusable ``aiosqlite`` connections.

    Connections are opened once (lazily on first use) and handed out via
    :meth:`acquire`; callers return them automatically when the context exits.

    Thread-safety scope (deliberate): this is safe for concurrent *coroutines*
    on a single asyncio event loop — what this design needs — because the
    backing ``asyncio.Queue``/``Lock`` are loop-bound. It is *not* OS-thread-safe
    for that same reason, and because a SQLite connection generally cannot be
    used from a thread other than the one that created it. To share across real
    OS threads, give each thread its own pool/loop rather than sharing one pool.

    SQLite write caveat: a bigger pool does NOT increase write throughput —
    SQLite serializes all writers behind a single database-level lock. The pool
    buys reuse (no per-write reconnect), concurrent reads, and a capped handle
    count. Each connection is opened with WAL (readers don't block the writer)
    and a busy_timeout (wait-and-retry instead of "database is locked").
    """

    def __init__(
        self,
        db_path: str,
        size: int = 5,
        timeout: float = 30.0,
        init_statements: list[str] | None = None,
    ) -> None:
        self.db_path = db_path
        self.size = max(1, size)
        self.timeout = timeout
        # Setup SQL run on every connection at open (e.g. schema DDL).
        self.init_statements = init_statements or []
        self._pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=self.size)
        self._all: list[aiosqlite.Connection] = []
        self._lock = asyncio.Lock()
        self._opened = False

    async def open(self) -> None:
        """Open ``size`` connections and run setup SQL on each. Idempotent."""
        async with self._lock:
            if self._opened:
                return
            for _ in range(self.size):
                conn = await aiosqlite.connect(self.db_path, timeout=self.timeout)
                # WAL: readers don't block the single writer. busy_timeout:
                # wait (don't error) when another connection holds the write lock.
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA busy_timeout=30000")
                for stmt in self.init_statements:
                    await conn.execute(stmt)
                await conn.commit()
                self._all.append(conn)
                self._pool.put_nowait(conn)
            self._opened = True

    def available(self) -> int:
        """Number of connections currently idle in the pool."""
        return self._pool.qsize()

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[aiosqlite.Connection]:
        """Borrow a connection, blocking if all are in use; auto-returns it."""
        if not self._opened:
            await self.open()
        conn = await self._pool.get()
        try:
            yield conn
        finally:
            self._pool.put_nowait(conn)

    async def close(self) -> None:
        """Close every pooled connection. Assumes none are currently acquired."""
        async with self._lock:
            if not self._opened:
                return
            while not self._pool.empty():
                self._pool.get_nowait()
            for conn in self._all:
                await conn.close()
            self._all.clear()
            self._opened = False


class TelemetryWriter:
    """Persist telemetry rows, borrowing a connection from a shared pool.

    Holds no connection of its own — it only references the pool — so it is
    cheap to create per worker and safe to use concurrently.
    """

    def __init__(self, pool: SqliteConnectionPool) -> None:
        self.pool = pool

    async def write(self, item: TelemetryData) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO telemetry (device_id, metric) VALUES (?, ?)",
                (item.device_id, str(item.metric)),
            )
            await conn.commit()


class IoWorker:
    """One asyncio worker's consume loop.

    Creates its own :class:`TelemetryWriter` over the *shared* connection pool,
    then pulls telemetry off the queue until a ``None`` sentinel arrives. Every
    ``get()`` is balanced by exactly one ``task_done()`` so ``io_queue.join()``
    unblocks correctly.
    """

    def __init__(self, io_queue: asyncio.Queue, worker_id: int, pool: SqliteConnectionPool) -> None:
        self.io_queue = io_queue
        self.worker_id = worker_id
        self.writer = TelemetryWriter(pool)

    async def run(self) -> None:
        print(f"IO worker {self.worker_id} started.")
        while True:
            item = await self.io_queue.get()
            try:
                if item is None:  # poison pill -> stop this worker
                    print(f"IO worker {self.worker_id} received sentinel; exiting.")
                    break

                # Validate BEFORE touching attributes (fixes the original crash).
                if not isinstance(item, TelemetryData):
                    print(f"IO worker {self.worker_id}: skipping invalid {item!r}")
                    continue

                await self.writer.write(item)
                print(f"IO worker {self.worker_id} stored telemetry for {item.device_id}.")
            finally:
                self.io_queue.task_done()


class IoWorkerPool:
    """Owns the asyncio queue, the connection pool, and the pool of workers."""

    def __init__(
        self,
        maxsize: int = 100,
        num_workers: int = 5,
        db_path: str = "telemetry.db",
        pool_size: int | None = None,
    ) -> None:
        # Bounded queue so a burst of producers can't exhaust memory.
        self.io_queue: asyncio.Queue = asyncio.Queue(maxsize)
        self.num_workers = num_workers

        # One shared, coroutine-safe pool; default one connection per worker.
        self.pool = SqliteConnectionPool(
            db_path,
            size=pool_size or max(1, num_workers),
            init_statements=[TELEMETRY_DDL],
        )

        # Workers start immediately; construction must happen inside a running
        # event loop (e.g. within an async function or test).
        self.io_workers = [
            IoWorker(self.io_queue, i, self.pool) for i in range(num_workers)
        ]
        self.workers = [asyncio.create_task(worker.run()) for worker in self.io_workers]

    async def insert_io_task(self, telemetry_data: TelemetryData) -> None:
        """Producer entry point. Awaits if the queue is full (backpressure)."""
        print(f"Inserting telemetry for device {telemetry_data.device_id}.")
        await self.io_queue.put(telemetry_data)

    async def shutdown(self) -> None:
        """Drain work, stop workers, await them, then close the pool.

        Order matters: wait for queued work to finish, THEN send one sentinel
        per worker, THEN await the worker tasks, THEN close the connections.
        """
        await self.io_queue.join()
        for _ in range(self.num_workers):
            await self.io_queue.put(None)
        await asyncio.gather(*self.workers)
        await self.pool.close()


async def main() -> None:
    io_worker_pool = IoWorkerPool()

    for i in range(100):
        metric = {"temperature": 20 + i, "humidity": 50 + i}
        await io_worker_pool.insert_io_task(
            TelemetryData(device_id=f"device_{i}", metric=metric)
        )

    await io_worker_pool.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
