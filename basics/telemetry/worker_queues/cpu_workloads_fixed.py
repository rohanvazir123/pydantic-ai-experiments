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
#   * ``CpuWorkerPool``          — owns the queues, the sinks, and the processes
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
#    instance whose only state is spawn-picklable (the queues, the sinks, an int
#    id, the manager proxies) — it never references the pool manager.

from __future__ import annotations

import multiprocessing
import os
import queue as queue_mod
import tempfile
from abc import ABC, abstractmethod
from multiprocessing import JoinableQueue, Process, Queue
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
    results go (the job does). Holds only spawn-picklable state (the queues, the
    sinks, an int id, the manager proxies), so its :meth:`run` method is a valid
    ``Process`` target under ``spawn``.
    """

    def __init__(
        self,
        task_queue: JoinableQueue,
        result_queue: Queue | None,
        sinks: Mapping[str, ResultSink],
        worker_id: int,
        status: dict,
        cancelled: dict,
    ) -> None:
        super().__init__(worker_id)
        self.task_queue = task_queue
        self.result_queue = result_queue

        # The result destinations (file, http, …). The worker holds them as
        # infrastructure and hands them to the job; the job picks which one.
        self.sinks = sinks

        # Manager-backed proxies shared with the parent (and the other workers),
        # so status/cancellation are visible across processes.
        self.status = status
        self.cancelled = cancelled

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

                self.status[job.job_id] = JobStatus.RUNNING
                result = job.process(self.sinks)  # job computes AND routes its result
                if self.result_queue is not None:
                    self.result_queue.put(result)
                self.status[job.job_id] = JobStatus.DONE
            finally:
                self.task_queue.task_done()  # balances every get(), incl. sentinel/invalid


class CpuWorkerPool(WorkerPool):
    """Owns the task/result queues, the result sinks, and the pool of processes.

    This is the natural **service layer** for a FastAPI app — a route handler
    just holds one pool and calls its methods, e.g.::

        pool = CpuWorkerPool()

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

    def __init__(
        self,
        num_workers: int | None = None,
        collect_results: bool = True,
        result_path: str | None = None,
    ) -> None:
        # Default to one worker per core; overridable for tests.
        super().__init__(num_workers or multiprocessing.cpu_count())

        # Task queue (parent -> workers) and optional result queue (workers -> parent).
        self.task_queue: JoinableQueue = JoinableQueue()
        self.result_queue: Queue | None = Queue() if collect_results else None

        # Job bookkeeping shared across processes via a Manager: workers run in
        # child processes, so a plain dict/set in the parent wouldn't be visible
        # to them. ``_status`` maps job_id -> JobStatus; ``_cancelled`` is a
        # set-like dict of flagged job_ids.
        self._manager = multiprocessing.Manager()
        self._status = self._manager.dict()
        self._cancelled = self._manager.dict()

        # Result sinks shared across all worker processes. Each needs Manager-
        # backed state so it's picklable to the children and their writes are
        # visible to the parent: the file sink takes a Manager lock (serialises
        # cross-process appends), the http sink an outbox list (stands in for
        # real POSTs). A job picks its sink by name via ``result_sink``.
        if result_path is None:
            fd, result_path = tempfile.mkstemp(prefix="cpu_results_", suffix=".jsonl")
            os.close(fd)
        self.result_path = result_path
        self.http_outbox = self._manager.list()
        self.sinks: dict[str, ResultSink] = {
            "file": FileWriter(self.result_path, self._manager.Lock()),
            "http": HttpResponder(self.http_outbox),
        }

        # Each worker is its own object; only worker state crosses to the child.
        self.workers = [
            CpuWorker(
                self.task_queue, self.result_queue, self.sinks, i, self._status, self._cancelled
            )
            for i in range(self.num_workers)
        ]
        self.processes = [Process(target=worker.run) for worker in self.workers]

        print(f"Starting {self.num_workers} CPU worker processes.")
        for process in self.processes:
            process.start()

    def insert_job(self, job: Job) -> None:
        """Submit one job and mark it ``QUEUED``. Blocks if the queue is full."""
        self._status[job.job_id] = JobStatus.QUEUED
        self.task_queue.put(job)

    def insert_cpu_tasks(self, raw_payloads: list[Job]) -> None:
        """Producer entry point for a batch; each is tracked via :meth:`insert_job`."""
        for payload in raw_payloads:
            self.insert_job(payload)

    def cancel_job(self, job_id: str) -> bool:
        """Lazily cancel a still-queued job (a worker skips it when reached)."""
        if self._status.get(job_id, JobStatus.UNKNOWN) != JobStatus.QUEUED:
            return False  # unknown, or already running/done/cancelled
        self._cancelled[job_id] = True
        self._status[job_id] = JobStatus.CANCELLED
        return True

    def get_job_status(self, job_id: str) -> JobStatus:
        """Current status of a job, or ``UNKNOWN`` if this pool never saw it."""
        return self._status.get(job_id, JobStatus.UNKNOWN)

    def join_tasks(self) -> None:
        """Block until every submitted task has been marked done."""
        self.task_queue.join()

    def collect_results(self, expected: int, timeout: float = 30.0) -> list[ProcessedImage]:
        """Drain exactly ``expected`` results. Call after :meth:`join_tasks`.

        Because each worker puts its result before calling ``task_done()``,
        once ``join_tasks()`` returns all results are guaranteed to be enqueued.
        """
        if self.result_queue is None:
            return []
        results: list[ProcessedImage] = []
        for _ in range(expected):
            results.append(self.result_queue.get(timeout=timeout))
        return results

    def shutdown(self) -> None:
        """Send one poison pill per worker, then join the processes.

        Sentinels are sent BEFORE joining (the original bug was doing it the
        other way round, which deadlocked).
        """
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        for process in self.processes:
            process.join()
        self._manager.shutdown()  # tear down the bookkeeping manager process

    def drain_extra_results(self) -> list[ProcessedImage]:
        """Best-effort drain of any results not consumed by ``collect_results``."""
        if self.result_queue is None:
            return []
        extra: list[ProcessedImage] = []
        while True:
            try:
                extra.append(self.result_queue.get_nowait())
            except queue_mod.Empty:
                break
        return extra


if __name__ == "__main__":
    pool = CpuWorkerPool(num_workers=4)

    # Even ids persist to the file sink; odd ids are "sent back" over the http sink.
    payloads = [
        ImageProcessingRequest(
            image_id=f"image_{i}",
            image_data=b"fake_image_data" * (i + 1),
            result_sink="file" if i % 2 == 0 else "http",
        )
        for i in range(20)
    ]
    pool.insert_cpu_tasks(payloads)

    # Wait for all real work to finish, collect results, THEN shut the pool down.
    pool.join_tasks()
    results = pool.collect_results(len(payloads))
    persisted = read_results(pool.result_path)
    sent = list(pool.http_outbox)
    pool.shutdown()

    print(f"Processed {len(results)} images.")
    print(f"  file sink persisted {len(persisted)} → {pool.result_path}")
    print(f"  http sink sent back  {len(sent)} results")
