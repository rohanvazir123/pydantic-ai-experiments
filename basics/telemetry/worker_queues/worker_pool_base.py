from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

# Shared shape for both worker pools (CPU/multiprocessing, IO/asyncio): a Job goes
# on a queue, N Workers consume it until a sentinel, under one WorkerPool.
# CpuWorker/CpuWorkerPool are sync (child processes); IoWorker/IoWorkerPool are
# async (coroutines on one event loop) -- same contract, two colours.


class JobStatus(StrEnum):
    """Lifecycle of a single job in a pool."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class Job(BaseModel):
    """A unit of work placed on a queue; processes itself via ``process()``."""

    job_id: str = Field(default_factory=lambda: uuid4().hex)
    type: str = ""

    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Do the work and return a result. Sync (CPU) or async (IO)."""
        raise NotImplementedError


class Worker(Protocol):
    """One worker's consume loop over a shared queue."""

    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id

    def start(self) -> None:
        """Create this worker's task/process and launch ``run()``."""
        raise NotImplementedError

    def run(self) -> Any:
        """Consume the queue until a sentinel. Sync (CPU) or async (IO)."""
        raise NotImplementedError


class WorkerPool(Protocol):
    """Owns the task queue and a fixed set of ``Worker``s."""

    _status: Mapping[str, JobStatus]

    def __init__(self, num_workers: int) -> None:
        self.num_workers = num_workers

    def start(self) -> None:
        """Build the pool: queue, shared state, sinks, workers."""
        self._init_queue()
        self._init_shared_state()
        self._init_sinks()
        self._init_workers()

    def _init_queue(self) -> None:
        raise NotImplementedError

    def _init_shared_state(self) -> None:
        raise NotImplementedError

    def _init_sinks(self) -> None:
        raise NotImplementedError

    def _init_workers(self) -> None:
        raise NotImplementedError

    def insert_job(self, job: Job) -> Any:
        """Submit one job; marks it QUEUED. Sync (CPU) or async (IO)."""
        raise NotImplementedError

    def cancel_job(self, job_id: str) -> bool:
        """Lazy cancel: flag a queued job so its worker skips it."""
        raise NotImplementedError

    def get_job_status(self, job_id: str) -> JobStatus:
        """Current status, or UNKNOWN if never seen."""
        raise NotImplementedError

    def join_tasks(self) -> Any:
        """Block until every submitted job has been processed."""
        raise NotImplementedError

    def collect_results(self) -> dict[str, JobStatus]:
        """Each job's outcome (job_id -> JobStatus). Call after ``join_tasks()``."""
        return dict(self._status)

    def shutdown(self) -> Any:
        """Stop all workers cleanly (sentinels first). Sync (CPU) or async (IO)."""
        raise NotImplementedError
