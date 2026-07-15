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
#
# One lifecycle rule holds at both levels: **construction is inert, ``start()``
# runs.** ``__init__`` only records config; ``WorkerPool.start()`` builds the
# queue/state/sinks/workers, and ``Worker.start()`` launches one worker's task or
# process. Setup and teardown are then both explicit calls (``start`` /
# ``shutdown``) rather than one being hidden in a constructor.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    """Lifecycle of a single job in a pool. ``StrEnum`` → JSON/proxy friendly."""

    QUEUED = "queued"      # accepted, waiting in the queue
    RUNNING = "running"    # a worker has picked it up
    DONE = "done"          # processed successfully
    FAILED = "failed"      # process() raised; recorded so the parent can react
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

    Construction is inert; :meth:`start` is what runs. That split is what lets a
    worker be built and inspected in a test without it draining the queue, and it
    keeps ``IoWorker`` constructible outside an event loop.
    """

    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id

    @abstractmethod
    def start(self) -> None:
        """Create this worker's own task (IO) / process (CPU) and launch :meth:`run`.

        The worker creates its handle rather than being handed one: the handle can
        only be built from ``self.run``, so it cannot exist before the worker does.

        The alternative — the pool creating it — would keep ``run()``
        execution-agnostic, leaving "task vs thread vs process" the pool's choice.
        That abstraction is unusable here: ``IoWorker.run`` is a coroutine awaiting
        an ``asyncio.Queue``, and ``CpuWorker`` is built around spawn-picklable
        state because it crosses a process boundary. Neither is portable to the
        other execution model, so the indirection would buy nothing. (It would be
        worth revisiting for structured concurrency — an ``asyncio.TaskGroup``
        owning the workers has to call ``create_task`` from the pool.)

        Each worker keeps its handle (``self.task`` / ``self.process``) so the pool
        can join it during ``shutdown``.
        """

    @abstractmethod
    def run(self) -> Any:
        """Consume the queue until a sentinel. Sync (CPU) or async (IO)."""


class WorkerPool(ABC):
    """Owns the task queue and a fixed set of :class:`Worker`s.

    Holds the shared ``num_workers`` and defines the lifecycle contract both
    pools implement: ``start()`` builds, ``shutdown()`` tears down.
    :meth:`shutdown` is **synchronous** in ``CpuWorkerPool`` and a **coroutine**
    in ``IoWorkerPool`` — same intent (stop every worker cleanly, sentinels
    before join), two execution models.
    """

    # Populated by subclasses (job_id -> JobStatus). Declared here so the shared
    # collect_results() can read it. CPU backs it with a Manager proxy, IO with a
    # plain dict -- both are Mappings.
    _status: Mapping[str, JobStatus]

    def __init__(self, num_workers: int) -> None:
        # Config only -- nothing is built and nothing runs. Subclasses record their
        # own config (queue maxsize, DB engine) around this call; start() reads it.
        self.num_workers = num_workers

    def start(self) -> None:
        """Build the pool: queue, shared state, sinks, workers. **Template method.**

        The ordered sequence both pools follow, declared once here. Each step is
        abstract because the two share the shape but no mechanics (``JoinableQueue``
        vs ``asyncio.Queue``; ``Manager`` proxies vs a plain dict).

        Separate from ``__init__`` because every step has a runtime side effect —
        ``_init_workers`` spawns processes (CPU) and needs a running event loop
        (IO). A constructor is the wrong home for those: it would call overridable
        methods on a half-built object, and a partially-constructed pool would
        escape if a step raised. Being a real method also means subclasses record
        their config normally instead of having to set it before ``super().__init__``.
        """
        self._init_queue()
        self._init_shared_state()
        self._init_sinks()
        self._init_workers()

    @abstractmethod
    def _init_queue(self) -> None:
        """Step 1: the queue carrying jobs parent -> workers."""

    @abstractmethod
    def _init_shared_state(self) -> None:
        """Step 2: the ``_status`` / ``_cancelled`` bookkeeping shared with the workers."""

    @abstractmethod
    def _init_sinks(self) -> None:
        """Step 3: where a finished job's PAYLOAD goes.

        A registry of destinations in ``CpuWorkerPool`` (file / http, picked per
        job); a single DB writer in ``IoWorkerPool``, whose only sink is the row it
        persists. Distinct from the OUTCOME (``_status``), which every pool reports
        the same way via :meth:`collect_results`.
        """

    @abstractmethod
    def _init_workers(self) -> None:
        """Step 4: build one :class:`Worker` per slot and ``start()`` each.

        The pool never creates a task or process itself — each worker does, in its
        own ``start()``. See :meth:`Worker.start`.
        """

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
    def join_tasks(self) -> Any:
        """Block until every submitted task has been processed.

        Sync (CPU) or async (IO). The result barrier: after this returns, every
        outcome is available to :meth:`collect_results`.
        """

    def collect_results(self) -> dict[str, JobStatus]:
        """Return each job's OUTCOME (job_id -> JobStatus). Call after join_tasks().

        The "result" of a fire-and-forget job is whether it succeeded, not its
        payload. The payload already lives in its sink -- a DB row, a file, an HTTP
        callback -- so we never read the sink back; if telemetry was persisted the
        job is DONE, and that's the result. Concrete and shared: identical for
        every pool, since both already maintain ``_status``. FAILED / stuck jobs
        show up here too, so the orchestrator can reconcile and take corrective
        action. Cheap (reads an in-memory map), so it stays synchronous.
        """
        return dict(self._status)

    @abstractmethod
    def shutdown(self) -> Any:
        """Stop all workers cleanly (sentinels first). Sync (CPU) or async (IO)."""
