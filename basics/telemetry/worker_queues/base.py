# =====================================================================
# Shared abstractions for the worker pools
# =====================================================================
#
# Both pools — CPU (multiprocessing) and IO (asyncio) — have the same shape:
# a typed message goes on a queue, N workers consume it until a sentinel. These
# three bases capture that shape so the two files share one vocabulary instead
# of each re-declaring it:
#
#   Job          (pydantic BaseModel) → the message on the queue; self-processing
#   Worker       (ABC)                → one worker's consume loop
#   WorkerPool   (ABC)                → owns the queue + N workers
#
# Concretely:
#   ImageProcessingRequest(Job)   CpuWorker(Worker)   CpuWorkerPool(WorkerPool)
#   TelemetryData(Job)            IoWorker(Worker)    IoWorkerPool(WorkerPool)
#
# Why Job is a pydantic model but Worker/WorkerPool are ABCs: a Job is
# *data* (validated, serialisable, picklable across the process boundary), so a
# BaseModel fits. A Worker/WorkerPool is *behaviour* holding non-serialisable
# runtime state (queues, child processes, asyncio tasks, DB connections) — that
# is not a pydantic model; an abstract base class is the right common type.
#
# The two implementations diverge on one axis the ABCs can't hide: the CPU side
# is synchronous (methods run in child processes), the IO side is asynchronous
# (coroutines on one event loop). So ``run`` / ``shutdown`` are sync in the CPU
# subclasses and ``async`` in the IO subclasses — same contract, two colours.

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    """Lifecycle of a single job in a pool. ``StrEnum`` → JSON/proxy friendly."""

    QUEUED = "queued"      # accepted, waiting in the queue
    RUNNING = "running"    # a worker has picked it up
    DONE = "done"          # processed successfully
    CANCELLED = "cancelled"  # skipped before processing (lazy cancel)
    UNKNOWN = "unknown"    # never seen by this pool


class Job(BaseModel):
    """The unit of work placed on a queue (merged base — was ``WorkItem`` + ``Job``).

    A compute job **processes itself**: subclasses override :meth:`process` to do
    the work and return a result, and the worker just calls ``job.process()``
    (Command pattern). So the worker holds no processor and branches on nothing —
    new job types = new subclasses, no worker change (open/closed).

    - ``job_id`` — auto-generated, so a pool can track and cancel it by id.
    - ``type`` — optional label / discriminator.
    """

    job_id: str = Field(default_factory=lambda: uuid4().hex)
    type: str = ""

    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Do this job's work and return a result. Override in concrete jobs.

        Same "two colours" split as ``run`` / ``shutdown``: a **CPU** job
        overrides ``process()`` as pure, synchronous compute (no args); an
        **IO** job overrides it as an ``async`` coroutine that takes the
        infrastructure resource it needs (e.g. a writer for a DB sink), since
        its work is I/O, not computation. Hence the ``*args`` here.
        """
        raise NotImplementedError


class Worker(ABC):
    """One worker's consume loop over a shared queue.

    Subclasses pull :class:`Job`s until a ``None`` sentinel, validate,
    process, and mark each task done. :meth:`run` is **synchronous** in
    ``CpuWorker`` (it runs inside a child process) and a **coroutine** in
    ``IoWorker`` (it runs on the event loop) — hence the ``Any`` return.
    """

    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id

    @abstractmethod
    def run(self) -> Any:
        """Consume the queue until a sentinel. Sync (CPU) or async (IO)."""


class WorkerPool(ABC):
    """Owns the task queue and a fixed set of :class:`Worker`s.

    Holds the shared ``num_workers`` and defines the lifecycle contract both
    pools implement. :meth:`shutdown` is **synchronous** in ``CpuWorkerPool``
    and a **coroutine** in ``IoWorkerPool`` — same intent (stop every worker
    cleanly, sentinels before join), two execution models.
    """

    def __init__(self, num_workers: int) -> None:
        self.num_workers = num_workers

    @abstractmethod
    def insert_job(self, job: Job) -> Any:
        """Submit one job; marks it ``QUEUED``. Sync (CPU) or async (IO)."""

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Best-effort **lazy** cancel: a queued job can't be pulled out of the
        queue, so it's flagged and the worker skips it when reached.

        Returns ``True`` if the job was still ``QUEUED`` and is now marked
        ``CANCELLED``; ``False`` if it is unknown or already running/done/cancelled.
        """

    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        """Current :class:`JobStatus`, or ``UNKNOWN`` if never seen."""

    @abstractmethod
    def shutdown(self) -> Any:
        """Stop all workers cleanly (sentinels first). Sync (CPU) or async (IO)."""
