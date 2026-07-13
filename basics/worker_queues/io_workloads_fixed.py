# =====================================================================
# IO-BOUND worker pool (asyncio: telemetry APIs, webhooks, DB logs)
# =====================================================================
#
# Fixed/reviewed version of ``io_workloads.py``. The original is left unchanged
# for reference; this file corrects the logic bugs.
#
# Bugs fixed vs the original
# --------------------------
# 1. Fake-async SQLite: the original did ``await sqlite3.connect(...)`` and
#    ``await cursor.execute(...)``. ``sqlite3`` is synchronous, so those awaits
#    raise ``TypeError``. This version uses ``aiosqlite`` for genuinely async
#    (non-blocking) database access.
# 2. Wrong ``task_done`` target: the original called ``queue.task_done()`` on
#    the imported ``queue`` *module* (and even ``await``ed it). Here it is
#    ``self.io_queue.task_done()``, called once per ``get()`` in a ``finally``.
# 3. ``insert_io_task`` is now a coroutine using ``await self.io_queue.put(...)``
#    so it applies backpressure at ``maxsize`` and can be awaited as the caller
#    (and the original ``main``) expected.
# 4. Removed the bogus ``from dbm import sqlite3`` and unused ``import io``.
# 5. Type validation happens before any attribute access.

import asyncio

import aiosqlite
from pydantic import BaseModel


class TelemetryData(BaseModel):
    device_id: str
    metric: dict


class IoWorkerQueue:
    """A pool of asyncio tasks consuming IO-bound telemetry writes."""

    def __init__(
        self,
        maxsize: int = 100,
        num_workers: int = 5,
        db_path: str = "telemetry.db",
    ) -> None:
        # Bounded queue so a burst of producers can't exhaust memory; workers
        # apply backpressure via ``await put`` once ``maxsize`` is reached.
        self.io_queue: asyncio.Queue = asyncio.Queue(maxsize)
        self.num_workers = num_workers
        self.db_path = db_path

        # Workers start immediately; construction must happen inside a running
        # event loop (e.g. within an async function or test).
        self.workers = [
            asyncio.create_task(self.process_io_work(i)) for i in range(num_workers)
        ]

    async def insert_io_task(self, telemetry_data: TelemetryData) -> None:
        """Producer entry point. Awaits if the queue is full (backpressure)."""
        print(f"Inserting telemetry for device {telemetry_data.device_id}.")
        await self.io_queue.put(telemetry_data)

    async def process_io_work(self, worker_id: int) -> None:
        """Consumer loop: pull telemetry and persist it until a sentinel arrives.

        Every ``get()`` is balanced by exactly one ``task_done()`` in the
        ``finally`` block, so ``io_queue.join()`` unblocks correctly.
        """
        print(f"IO worker {worker_id} started.")
        while True:
            item = await self.io_queue.get()
            try:
                if item is None:  # poison pill -> stop this worker
                    print(f"IO worker {worker_id} received sentinel; exiting.")
                    break

                # Validate BEFORE touching attributes (fixes the original crash).
                if not isinstance(item, TelemetryData):
                    print(f"IO worker {worker_id}: skipping invalid item {item!r}")
                    continue

                await self._write_telemetry(item)
                print(f"IO worker {worker_id} stored telemetry for {item.device_id}.")
            finally:
                self.io_queue.task_done()

    async def _write_telemetry(self, item: TelemetryData) -> None:
        """Persist one telemetry row using genuinely async SQLite."""
        # ``timeout`` reduces "database is locked" errors when several workers
        # write to the same file concurrently.
        async with aiosqlite.connect(self.db_path, timeout=30) as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS telemetry (device_id TEXT, metric TEXT)"
            )
            await db.execute(
                "INSERT INTO telemetry (device_id, metric) VALUES (?, ?)",
                (item.device_id, str(item.metric)),
            )
            await db.commit()

    async def shutdown(self) -> None:
        """Drain outstanding work, stop workers, and await their completion.

        Order matters: wait for queued work to finish, THEN send one sentinel
        per worker, THEN await the worker tasks.
        """
        await self.io_queue.join()
        for _ in range(self.num_workers):
            await self.io_queue.put(None)
        await asyncio.gather(*self.workers)


async def main() -> None:
    io_worker_queue = IoWorkerQueue()

    for i in range(100):
        metric = {"temperature": 20 + i, "humidity": 50 + i}
        await io_worker_queue.insert_io_task(
            TelemetryData(device_id=f"device_{i}", metric=metric)
        )

    await io_worker_queue.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
