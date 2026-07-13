# =====================================================================
# IO-BOUND worker pool (asyncio) with a SQLAlchemy sink
# =====================================================================
#
# Persistence goes through SQLAlchemy Core against an in-memory SQLite DB.
# SQLAlchemy is DB-agnostic, so this is the "work against a scalable database"
# answer: swap the engine URL for ``postgresql://...`` and the pool/writer code
# is unchanged. SQLAlchemy also manages connection pooling, so there's no
# hand-rolled connection pool here.
#
# SQLite specifics: ``:memory:`` is per-connection, so we use ``StaticPool`` (one
# shared connection) + ``check_same_thread=False`` so the ``asyncio.to_thread``
# worker threads all see the same DB. A ``threading.Lock`` serialises writes
# (SQLite serialises writers anyway; on Postgres you'd drop both and use a real
# pool). ``to_thread`` keeps the blocking driver off the event loop.

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from base import Job, JobStatus, Worker, WorkerPool
from sqlalchemy import (
    Column,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    insert,
    select,
)
from sqlalchemy.pool import StaticPool

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


class TelemetryData(Job):
    device_id: str
    metric: dict

    async def process(self, writer: TelemetryWriter) -> None:
        """Do this job's work: persist itself through the given writer.

        Mirrors the CPU job's ``process()`` (the job does its own work), but an
        IO job's work is a DB write, so it needs the writer/engine the pool owns
        — passed in rather than held. Async because the write is off-loop.
        """
        await writer.write(self)


_metadata = MetaData()
telemetry = Table(
    "telemetry",
    _metadata,
    Column("device_id", String),
    Column("metric", String),
)


def make_engine(url: str = "sqlite:///:memory:") -> Engine:
    """Build the engine + schema. Swap ``url`` for ``postgresql://...`` in prod."""
    engine = create_engine(
        url, connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    _metadata.create_all(engine)
    return engine


def count_rows(engine: Engine) -> int:
    """Helper: how many telemetry rows are persisted."""
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(telemetry)).scalar_one()


class TelemetryWriter:
    """Persist telemetry via SQLAlchemy, off the event loop."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._lock = threading.Lock()  # serialise the shared in-memory connection

    async def write(self, item: TelemetryData) -> None:
        await asyncio.to_thread(self._write, item)

    def _write(self, item: TelemetryData) -> None:
        with self._lock, self.engine.begin() as conn:
            conn.execute(
                insert(telemetry).values(
                    device_id=item.device_id, metric=str(item.metric)
                )
            )


class IoWorker(Worker):
    """One asyncio worker: pull items until a sentinel, let each job persist itself.

    The worker holds the writer as an infrastructure resource (the DB
    connection) and hands it to ``item.process(self.writer)`` — it branches on
    nothing and knows no SQL, mirroring ``CpuWorker`` calling ``job.process()``.
    """

    def __init__(
        self,
        io_queue: asyncio.Queue,
        worker_id: int,
        writer: TelemetryWriter,
        status: dict[str, JobStatus],
        cancelled: set[str],
    ) -> None:
        super().__init__(worker_id)
        self.io_queue = io_queue
        self.writer = writer
        self.status = status
        self.cancelled = cancelled

    async def run(self) -> None:
        while True:
            item = await self.io_queue.get()
            try:
                if item is None:  # poison pill
                    break
                if not isinstance(item, TelemetryData):
                    continue
                if item.job_id in self.cancelled:  # lazy cancel
                    self.status[item.job_id] = JobStatus.CANCELLED
                    continue
                self.status[item.job_id] = JobStatus.RUNNING
                await item.process(self.writer)  # the job does its own work
                self.status[item.job_id] = JobStatus.DONE
            finally:
                self.io_queue.task_done()


class IoWorkerPool(WorkerPool):
    """Owns the asyncio queue, the SQLAlchemy writer, and the pool of workers."""

    def __init__(
        self,
        engine: Engine | None = None,
        maxsize: int = 100,
        num_workers: int = 5,
    ) -> None:
        super().__init__(num_workers)
        self.engine = engine or make_engine()
        self.writer = TelemetryWriter(self.engine)
        self.io_queue: asyncio.Queue = asyncio.Queue(maxsize)
        self._status: dict[str, JobStatus] = {}
        self._cancelled: set[str] = set()
        self.io_workers = [
            IoWorker(self.io_queue, i, self.writer, self._status, self._cancelled)
            for i in range(num_workers)
        ]
        self.workers = [asyncio.create_task(w.run()) for w in self.io_workers]

    async def insert_job(self, job: TelemetryData) -> None:
        """Submit one job; mark ``QUEUED``; await if the queue is full."""
        self._status[job.job_id] = JobStatus.QUEUED
        await self.io_queue.put(job)

    def cancel_job(self, job_id: str) -> bool:
        """Lazily cancel a still-queued job (worker skips it when reached)."""
        if self._status.get(job_id, JobStatus.UNKNOWN) != JobStatus.QUEUED:
            return False
        self._cancelled.add(job_id)
        self._status[job_id] = JobStatus.CANCELLED
        return True

    def get_job_status(self, job_id: str) -> JobStatus:
        return self._status.get(job_id, JobStatus.UNKNOWN)

    async def shutdown(self) -> None:
        """Drain, then sentinel each worker, then await them (sentinels first)."""
        await self.io_queue.join()
        for _ in range(self.num_workers):
            await self.io_queue.put(None)
        await asyncio.gather(*self.workers)


async def main() -> None:
    pool = IoWorkerPool(num_workers=3)
    for i in range(10):
        await pool.insert_job(
            TelemetryData(device_id=f"device_{i}", metric={"temp": 20 + i})
        )
    await pool.shutdown()
    print(f"wrote {count_rows(pool.engine)} rows via SQLAlchemy")


if __name__ == "__main__":
    asyncio.run(main())
