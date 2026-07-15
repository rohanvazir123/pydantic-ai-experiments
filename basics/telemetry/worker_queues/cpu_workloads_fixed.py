# =====================================================================
# CPU-BOUND worker pool (multiprocessing: parsing, aggregating, image work)
# =====================================================================
#
# Fixed/reviewed version of ``cpu_workloads.py``. The original is left
# unchanged for reference; this file corrects the logic bugs and makes the
# pool safe under the ``spawn`` start method (macOS/Windows default).
#
# Responsibilities:
#   * ``ImageProcessingRequest`` — a self-processing CPU job (``process(sinks)``)
#   * ``ResultSink`` / ``FileWriter`` / ``HttpResponder`` — where a result goes
#   * ``CpuWorker``              — one worker process's consume loop
#   * ``CpuWorkerPool``          — owns the task queue, the sinks, and the processes
#
# FastAPI-ready: the pool is the service layer. ``insert_job`` / ``get_job_status``
# / ``cancel_job`` map 1:1 onto ``POST /jobs`` (returns ``job_id``), ``GET
# /jobs/{id}`` (poll ``JobStatus``), and ``DELETE /jobs/{id}``. Because every
# ``Job`` is a pydantic model it doubles as the request body, and the ``http``
# sink is exactly how a result gets pushed back to the caller (webhook/callback)
# instead of polled. See the note on ``CpuWorkerPool``.
#
# Bugs fixed vs the original
# --------------------------
# 1. Deadlock: the original joined the worker processes *before* sending the
#    poison-pill sentinels, so the workers (which only exit on a sentinel)
#    never terminated. Here ``shutdown()`` sends one sentinel per worker first,
#    then joins.
# 2. Attribute access before validation: the original read ``payload.image_id``
#    before checking ``isinstance(...)``. Here the type check happens first.
# 3. Spawn safety: the original used ``target=self.process_task``, which pickles
#    the whole *manager* instance — including its list of ``Process`` objects,
#    which cannot be pickled. Here the ``Process`` target is a ``CpuWorker``
#    instance whose only state is spawn-picklable (the task queue, the sinks, an
#    int id, the manager proxies) — it never references the pool manager.

from __future__ import annotations

import multiprocessing
import os
import tempfile
from abc import ABC, abstractmethod
from multiprocessing import JoinableQueue, Process
from typing import TYPE_CHECKING, Any

from base import Job, JobStatus, Worker, WorkerPool
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Mapping


class ProcessedImage(BaseModel):
    image_id: str
    size_bytes: int


# ---------------------------------------------------------------------------
# Result sinks: where a job's result goes is a property of the *job*, not the
# worker. The worker holds a registry of sinks and the job routes its own result
# into the one it wants (``result_sink``) — so one job type persists to a file
# and another is sent back over HTTP, with no branching in the worker.
# ---------------------------------------------------------------------------


class ResultSink(ABC):
    """A destination for a finished result. Chosen per job via ``result_sink``."""

    @abstractmethod
    def emit(self, result: BaseModel) -> None:
        """Persist / send one result. Must be spawn-picklable (runs in a child)."""


class FileWriter(ResultSink):
    """Durable sink: append each result as one JSON line.

    Workers run in **separate processes**, so appends to the shared file are
    serialised by a ``Manager`` lock — the cross-process analog of the IO
    writer's ``threading.Lock`` (many processes here vs many threads there).
    """

    def __init__(self, path: str, lock: Any) -> None:
        self.path = path
        self._lock = lock

    def emit(self, result: BaseModel) -> None:
        line = result.model_dump_json()
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")


class HttpResponder(ResultSink):
    """"Send it back over HTTP" sink — POST the result to a callback URL.

    Stubbed as an append to a shared ``Manager`` list (an outbox) so the demo
    stays dependency-free and testable; in prod ``emit`` would POST
    ``result.model_dump_json()`` to the job's callback URL.
    """

    def __init__(self, outbox: Any) -> None:
        self.outbox = outbox

    def emit(self, result: BaseModel) -> None:
        self.outbox.append(result.model_dump_json())


def read_results(path: str) -> list[ProcessedImage]:
    """Read back everything a :class:`FileWriter` persisted (one per line)."""
    with open(path) as f:
        return [ProcessedImage.model_validate_json(line) for line in f if line.strip()]


class ImageProcessingRequest(Job):
    """A self-processing CPU job: it does its own image work *and* routes its result.

    ``process(sinks)`` computes the result, then emits it into the sink named by
    ``result_sink`` (``"file"`` to persist, ``"http"`` to send back). The worker
    just calls ``job.process(self.sinks)`` — no injected processor, no branching.
    Subclass and override :meth:`process` to change the behaviour.
    """

    image_id: str
    image_data: bytes
    result_sink: str = "file"  # which sink in the registry handles the result

    def process(self, sinks: Mapping[str, ResultSink] | None = None) -> ProcessedImage:
        # Placeholder for real image work (decode/resize/filter…).
        result = ProcessedImage(image_id=self.image_id, size_bytes=len(self.image_data))
        if sinks is not None:  # None → pure compute (unit-testable without a sink)
            sinks[self.result_sink].emit(result)
        return result


class CpuWorker(Worker):
    """One worker process's consume loop.

    Generic: it pulls a :class:`Job` and calls ``job.process(self.sinks)`` — it
    holds no processor, knows no specific job types, and doesn't decide where
    results go (the job does). Holds only spawn-picklable state (the task queue,
    the sinks, an int id, the manager proxies), so its :meth:`run` method is a
    valid ``Process`` target under ``spawn``.
    """

    process: Process  # this worker's own child process; set by start(), joined at shutdown

    def __init__(
        self,
        task_queue: JoinableQueue,
        sinks: Mapping[str, ResultSink],
        worker_id: int,
        status: dict,
        cancelled: dict,
    ) -> None:
        super().__init__(worker_id)
        # task_queue: where jobs arrive. There's no result queue -- a job's payload
        # goes to its sink; its outcome (DONE/FAILED) goes to self.status.
        self.task_queue = task_queue

        # The result destinations (file, http, …). The worker holds them as
        # infrastructure and hands them to the job; the job picks which one.
        self.sinks = sinks

        # Manager-backed proxies shared with the parent (and the other workers),
        # so status/cancellation are visible across processes.
        self.status = status
        self.cancelled = cancelled

    def start(self) -> None:
        """Launch this worker's run loop in its own child process.

        The handle is bound to ``self`` only AFTER ``process.start()``: under
        ``spawn`` that call pickles the target ``self.run`` -- and therefore
        ``self`` -- into the child, so assigning first would ship the child a
        stale copy of its own Process object. Building the Process is inert;
        starting it is the side effect, which is why this isn't in ``__init__``.
        """
        process = Process(target=self.run)
        process.start()
        self.process = process

    def run(self) -> None:
        """Pull jobs until a ``None`` sentinel; balance every get with a done."""
        print(f"CPU worker {self.worker_id} started.")
        while True:
            job = self.task_queue.get()
            try:
                if job is None:  # poison pill -> shut this worker down
                    break

                # Validate BEFORE touching any attributes (fixes the original crash).
                if not isinstance(job, Job):
                    print(f"CPU worker {self.worker_id}: skipping invalid {job!r}")
                    continue

                # Lazy cancellation: a job flagged before we reached it is skipped.
                if job.job_id in self.cancelled:
                    self.status[job.job_id] = JobStatus.CANCELLED
                    print(f"CPU worker {self.worker_id}: skipping cancelled {job.job_id}")
                    continue

                # Mark running, do the work (the job computes AND routes its own
                # result into its chosen sink), then mark done. The payload leaves
                # via the sink; only the outcome is recorded here.
                self.status[job.job_id] = JobStatus.RUNNING
                try:
                    job.process(self.sinks)
                except Exception as exc:
                    # Contain the failure to THIS job. Without this except an
                    # unhandled error would propagate out of run() and kill the
                    # whole worker process -- shrinking the pool and, if enough
                    # workers die, hanging task_queue.join(). Instead we record
                    # FAILED (visible to the parent via get_job_status) and keep
                    # looping. We deliberately do NOT retry here; recovery policy
                    # is the parent/orchestrator's call (see collect_results).
                    self.status[job.job_id] = JobStatus.FAILED
                    print(f"CPU worker {self.worker_id}: job {job.job_id} failed: {exc!r}")
                    continue
                self.status[job.job_id] = JobStatus.DONE
            except Exception as exc:
                # Safety net around the WHOLE loop body: the tight except above
                # only covers job.process(). Anything else that can raise --
                # status bookkeeping, a manager-proxy hiccup -- must NOT kill the
                # worker either. Log and fall through to
                # `finally`; the while loop then keeps consuming. `break` (poison
                # pill) is not an exception, so shutdown still works.
                print(f"CPU worker {self.worker_id}: unexpected loop error: {exc!r}")
            finally:
                # Balance EVERY get() with a task_done() -- including the poison
                # pill and invalid jobs -- or task_queue.join() would block forever.
                self.task_queue.task_done()


class CpuWorkerPool(WorkerPool):
    """Owns the task queue, the result sinks, and the pool of processes.

    This is the natural **service layer** for a FastAPI app — a route handler
    just holds one pool and calls its methods, e.g.::

        pool = CpuWorkerPool()
        pool.start()                  # build queue/state/sinks/workers (on startup)

        @app.post("/jobs")            # body IS the pydantic Job
        def submit(job: ImageProcessingRequest) -> dict[str, str]:
            pool.insert_job(job)      # returns immediately (async work)
            return {"job_id": job.job_id}

        @app.get("/jobs/{job_id}")    # poll status
        def status(job_id: str) -> dict[str, str]:
            return {"status": pool.get_job_status(job_id)}

        @app.delete("/jobs/{job_id}")
        def cancel(job_id: str) -> dict[str, bool]:
            return {"cancelled": pool.cancel_job(job_id)}

    The result comes back one of two ways, chosen per job by ``result_sink``:
    the ``http`` sink POSTs it to the caller's callback URL (fire-and-forget),
    or the ``file`` sink persists it for a later ``GET``. Nothing here is tied to
    a web framework, so the same pool works under FastAPI, a CLI, or a consumer.
    """

    def __init__(self, num_workers: int | None = None) -> None:
        # Default to one worker per core; overridable for tests. Config only --
        # nothing is built and nothing runs here; that's WorkerPool.start().
        super().__init__(num_workers or multiprocessing.cpu_count())

    # -- Build steps, run in order by WorkerPool.start() ---------------------
    #
    # Workflow this pool wires up  (parent = orchestrator):
    #
    #   parent.insert_job(job)
    #        │
    #        ▼
    #   task_queue ───► worker[0..N] ───► job.process(sinks)
    #  (shared input)    (child procs)          │
    #                                           ├──► status[job_id] = DONE/FAILED
    #                                           │        (the OUTCOME; collect_results reads it)
    #                                           │
    #                                           └──► sinks[result_sink].emit()   (the PAYLOAD)
    #                                                     ├─► FileWriter  → disk    (durable; later GET)
    #                                                     └─► HttpResponder → webhook (push to caller)
    #
    # No result queue: the payload goes to its sink, the outcome goes to
    # status. The parent never has payloads pushed back to it.

    def _init_queue(self) -> None:
        """Step 1: wire the parent to the children with the task queue.

        ``task_queue`` carries jobs parent -> workers. A ``JoinableQueue`` so
        ``join_tasks()`` can block until every ``put()`` has a matching
        ``task_done()``.
        """
        self.task_queue: JoinableQueue = JoinableQueue()

    def _init_shared_state(self) -> None:
        """Step 2: create the cross-process job bookkeeping via a ``Manager``.

        The Manager proxies are SHARED STATE BETWEEN PROCESSES -- conceptually
        like inter-process shared memory: every process (parent + all workers)
        reads/writes one common ``_status`` / ``_cancelled``. Mechanically it's not
        real shared memory though: the Manager is a SEPARATE server process that
        owns the real dicts, and each proxy forwards reads/writes to it over IPC
        (pickle -> socket -> unpickle). Needed because workers run in child
        processes -- a plain dict in the parent would not be visible to them (no
        shared address space under ``spawn``).

        * ``_status``    job_id -> JobStatus  (workers write, parent reads on GET)
        * ``_cancelled`` set-like dict of flagged job_ids (parent sets, workers check)
        """
        self._manager = multiprocessing.Manager()
        self._status = self._manager.dict()
        self._cancelled = self._manager.dict()

    def _init_sinks(self) -> None:
        """Step 3: build the result-destination registry shared across worker processes.

        Each sink needs Manager-backed state so it's picklable to the children and
        its writes are visible to the parent: the file sink takes a Manager lock
        (serialises cross-process appends), the http sink an outbox list (stands in
        for real POSTs). A job picks its sink by name via ``result_sink``.
        """
        # A fresh temp file backs the durable "file" sink.
        fd, self.result_path = tempfile.mkstemp(prefix="cpu_results_", suffix=".jsonl")
        os.close(fd)
        self.http_outbox = self._manager.list()
        self.sinks: dict[str, ResultSink    ] = {
            "file": FileWriter(self.result_path, self._manager.Lock()),
            "http": HttpResponder(self.http_outbox),
        }

    def _init_workers(self) -> None:
        """Step 4: build one ``CpuWorker`` per slot; each launches its own process.

        Each worker is its own object; only spawn-picklable worker state crosses
        to the child (the task queue, the sinks, an int id, the manager proxies).
        """
        # Every worker shares the SAME queue, sinks, and manager proxies -- only
        # the worker id differs. Sharing is the point: one input queue fans jobs
        # out across workers, and one set of proxies gives a single coherent view
        # of status/cancellation across all processes.
        self.workers = [
            CpuWorker(
                self.task_queue,    # SHARED input  : all workers pull jobs from this one queue
                self.sinks,         # SHARED output : per-job payload delivery (file/http)
                i,                  # this worker's own id -- the only non-shared arg
                self._status,       # SHARED state  : job_id -> JobStatus (manager proxy)
                self._cancelled,    # SHARED state  : cancelled job_ids     (manager proxy)
            )
            for i in range(self.num_workers)
        ]

        print(f"Starting {self.num_workers} CPU worker processes.")
        for worker in self.workers:
            worker.start()  # the worker creates its own process; the pool never does

    # -- Public API ---------------------------------------------------------
    # These methods exist because the work is LONG-RUNNING and arrives over a
    # REST submit endpoint. The HTTP request can't block until an image is done,
    # so `POST /jobs` -> insert_job() returns a job_id immediately (fire-and-
    # forget), and the client/dashboard later polls `GET /jobs/{id}` ->
    # get_job_status() for progress. Because the submitting request is long gone
    # by the time work finishes, the PARENT is the only thing still around to
    # gather outcomes and relay them onward -- which is why collecting results
    # (and reporting status back to the client/dashboard) is the parent's job.

    def insert_job(self, job: Job) -> None:
        """Submit one job and mark it ``QUEUED``. Blocks if the queue is full.

        Backs `POST /jobs`: enqueue, then return the job_id to the caller at once
        instead of waiting for the (long-running) work to complete.
        """
        self._status[job.job_id] = JobStatus.QUEUED
        self.task_queue.put(job)

    def cancel_job(self, job_id: str) -> bool:
        """Lazily cancel a still-queued job (a worker skips it when reached)."""
        if self._status.get(job_id, JobStatus.UNKNOWN) != JobStatus.QUEUED:
            return False  # unknown, or already running/done/cancelled
        self._cancelled[job_id] = True
        self._status[job_id] = JobStatus.CANCELLED
        return True

    def get_job_status(self, job_id: str) -> JobStatus:
        """Current status of a job, or ``UNKNOWN`` if this pool never saw it.

        Backs `GET /jobs/{id}`: the endpoint the client/dashboard polls to render
        live progress (QUEUED -> RUNNING -> DONE / FAILED / CANCELLED).
        """
        return self._status.get(job_id, JobStatus.UNKNOWN)

    def join_tasks(self) -> None:
        """Block until every submitted task has been marked done."""
        self.task_queue.join()

    # collect_results() is inherited from WorkerPool: it reports each job's OUTCOME
    # (DONE / FAILED / ...) from _status. The actual ProcessedImage payload is
    # delivered by the job's sink (file / http), so we don't gather payloads here.

    def shutdown(self) -> None:
        """Send one poison pill per worker, then join the processes.

        Sentinels are sent BEFORE joining (the original bug was doing it the
        other way round, which deadlocked).
        """
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        for worker in self.workers:
            worker.process.join()
        self._manager.shutdown()  # tear down the bookkeeping manager process


if __name__ == "__main__":
    pool = CpuWorkerPool(num_workers=4)
    pool.start()  # build the queue, state, sinks, and workers

    # Even ids persist to the file sink; odd ids are "sent back" over the http sink.
    payloads = [
        ImageProcessingRequest(
            image_id=f"image_{i}",
            image_data=b"fake_image_data" * (i + 1),
            result_sink="file" if i % 2 == 0 else "http",
        )
        for i in range(20)
    ]
    for payload in payloads:
        pool.insert_job(payload)

    # Wait for all work to finish, read outcomes, THEN shut the pool down.
    pool.join_tasks()
    outcomes = pool.collect_results()  # job_id -> DONE / FAILED / ...
    persisted = read_results(pool.result_path)
    sent = list(pool.http_outbox)
    pool.shutdown()

    done = sum(1 for status in outcomes.values() if status == JobStatus.DONE)
    print(f"Processed {done}/{len(outcomes)} images successfully.")
    print(f"  file sink persisted {len(persisted)} → {pool.result_path}")
    print(f"  http sink sent back  {len(sent)} results")
