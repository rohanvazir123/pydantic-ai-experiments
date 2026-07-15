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

    task: asyncio.Task  # this worker's own run loop; set by start(), joined at shutdown

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

    def start(self) -> None:
        """Launch this worker's run loop as an asyncio task the worker owns.

        Needs a running event loop, and there is no way to build a task without
        also scheduling it -- which is exactly why this isn't in ``__init__``:
        constructing an ``IoWorker`` would otherwise start it. ``self.task`` is
        also the strong reference that keeps the loop from being GC'd mid-flight.
        """
        self.task = asyncio.create_task(self.run())

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
                try:
                    await item.process(self.writer)  # the job does its own work
                except Exception as exc:
                    # Contain the failure to THIS job. A DB write can fail
                    # (constraint violation, disconnect, timeout); without this
                    # except the error would bubble out of run() and kill the
                    # worker task -- shrinking the pool and potentially hanging
                    # io_queue.join(). We record FAILED (visible via
                    # get_job_status) and keep looping; retry/recovery policy is
                    # the parent/orchestrator's call. `except Exception` (not bare
                    # `except`) deliberately lets asyncio.CancelledError -- a
                    # BaseException -- propagate so shutdown still cancels cleanly.
                    self.status[item.job_id] = JobStatus.FAILED
                    print(f"IO worker {self.worker_id}: job {item.job_id} failed: {exc!r}")
                    continue
                self.status[item.job_id] = JobStatus.DONE
            except Exception as exc:
                # Safety net around the WHOLE loop body: the tight except above
                # only covers item.process(). Anything else that can raise --
                # status bookkeeping, queue accounting -- must NOT kill the worker
                # task either. Log and fall through to `finally`; the while loop
                # then keeps consuming. CancelledError is a BaseException, so
                # shutdown still cancels this task cleanly.
                print(f"IO worker {self.worker_id}: unexpected loop error: {exc!r}")
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
        # Config only -- nothing is built and nothing runs here. The four build
        # steps below are sequenced by WorkerPool.start(), which reads these two.
        self._maxsize = maxsize
        self._engine = engine
        super().__init__(num_workers)

    # -- Build steps, run in order by WorkerPool.start() ---------------------
    #
    # Workflow this pool wires up  (parent = orchestrator):
    #
    #   parent.insert_job(job)
    #        │
    #        ▼
    #   io_queue ───► worker[0..N] ───► job.process(writer)
    #  (shared input)  (async tasks)          │
    #                                          └──► TelemetryWriter.write() ──► DB
    #                                                 (asyncio.to_thread; off-loop)
    #
    # No result queue (unlike the CPU pool): an IO job's "result" is a row in
    # the database, so the writer/DB IS the sink. The parent tracks progress
    # via get_job_status and the outcome is the persisted row.

    def _init_queue(self) -> None:
        """Step 1: the single asyncio queue: parent -> workers, with backpressure at
        ``maxsize`` (``insert_job`` awaits when the queue is full)."""
        self.io_queue: asyncio.Queue = asyncio.Queue(self._maxsize)

    def _init_shared_state(self) -> None:
        """Step 2: job bookkeeping. Unlike the CPU pool, NO ``Manager`` is needed: asyncio
        runs one process on one event-loop thread, so a plain dict/set is safe --
        the only concurrency is cooperative (at ``await`` points), and this shared
        state is never touched inside ``asyncio.to_thread`` (only the DB write is).

        * ``_status``    job_id -> JobStatus  (workers write, parent reads on GET)
        * ``_cancelled`` set of flagged job_ids (parent adds, workers check)
        """
        self._status: dict[str, JobStatus] = {}
        self._cancelled: set[str] = set()

    def _init_sinks(self) -> None:
        """Step 3: the SQLAlchemy sink shared by all workers: engine + writer.

        Where the CPU pool builds a registry of destinations, this pool has exactly
        one: an IO job's payload IS the row it persists, so the DB is the only place
        a result can go. Swap the engine URL for ``postgresql://...`` in prod; the
        pool/writer are unchanged.
        """
        self.engine = self._engine or make_engine()
        self.writer = TelemetryWriter(self.engine)

    def _init_workers(self) -> None:
        """Step 4: build one ``IoWorker`` per slot; each launches its own task.

        Every worker shares the SAME queue, writer, and bookkeeping -- only the
        worker id differs. Requires a running event loop: the workers' ``start()``
        calls ``asyncio.create_task``.
        """
        self.workers = [
            IoWorker(self.io_queue, i, self.writer, self._status, self._cancelled)
            for i in range(self.num_workers)
        ]
        for worker in self.workers:
            worker.start()  # the worker creates its own task; the pool never does

    # -- Public API ---------------------------------------------------------
    # Same shape as CpuWorkerPool: long-running work arrives over a REST submit
    # endpoint. `POST /jobs` -> insert_job() returns a job_id immediately; the
    # client/dashboard polls `GET /jobs/{id}` -> get_job_status() for progress;
    # `DELETE /jobs/{id}` -> cancel_job(). The submitting request is long gone by
    # the time the DB write finishes, so the parent is what remains to track
    # outcomes and relay status back to the client/dashboard.

    async def insert_job(self, job: TelemetryData) -> None:
        """Submit one job and mark it ``QUEUED``; await if the queue is full.

        Backs `POST /jobs`: enqueue, then return the job_id to the caller at once
        instead of waiting for the (long-running) DB write to complete.
        """
        self._status[job.job_id] = JobStatus.QUEUED
        await self.io_queue.put(job)

    def cancel_job(self, job_id: str) -> bool:
        """Lazily cancel a still-queued job (a worker skips it when reached).

        Backs `DELETE /jobs/{id}`.
        """
        if self._status.get(job_id, JobStatus.UNKNOWN) != JobStatus.QUEUED:
            return False
        self._cancelled.add(job_id)
        self._status[job_id] = JobStatus.CANCELLED
        return True

    def get_job_status(self, job_id: str) -> JobStatus:
        """Current status of a job, or ``UNKNOWN`` if this pool never saw it.

        Backs `GET /jobs/{id}`: the endpoint the client/dashboard polls to render
        live progress (QUEUED -> RUNNING -> DONE / FAILED / CANCELLED).
        """
        return self._status.get(job_id, JobStatus.UNKNOWN)

    async def join_tasks(self) -> None:
        """Block until every submitted task has been processed.

        The result barrier: after this returns, every write has landed in the DB,
        so :meth:`collect_results` will see them all.
        """
        await self.io_queue.join()

    # collect_results() is inherited from WorkerPool: an IO job's "result" is
    # simply whether its write succeeded (DONE) -- reported from _status. We do
    # NOT read the persisted rows back; if telemetry landed, the job is DONE.

    async def shutdown(self) -> None:
        """Sentinel each worker, then await them (sentinels sent before the await).

        Call :meth:`join_tasks` first if you need every task drained before
        teardown; ``shutdown`` only stops the workers.
        """
        for _ in range(self.num_workers):
            await self.io_queue.put(None)
        await asyncio.gather(*(worker.task for worker in self.workers))


async def main() -> None:

    # Ceate a pool of IO workers, then build it (queue, state, sink, workers)
    pool = IoWorkerPool(num_workers=3)
    pool.start()

    # Submit multiple IO jobs to the pool. Each job is a TelemetryData instance
    for i in range(10):
        await pool.insert_job(
            TelemetryData(device_id=f"device_{i}", metric={"temp": 20 + i})
        )

    # Wait for all jobs to finish (the DB writes to land) before shutdown. 
    await pool.join_tasks()

    # Shutdown the pool cleanly (sentinels + await workers)
    await pool.shutdown()

    # Check how many jobs ran successfully
    outcomes = pool.collect_results()
    done = sum(1 for status in outcomes.values() if status == JobStatus.DONE)
    print(f"{done}/{len(outcomes)} writes succeeded; {count_rows(pool.engine)} rows in DB")


if __name__ == "__main__":
    asyncio.run(main())
