# =====================================================================
# CPU-BOUND worker pool (multiprocessing: parsing, aggregating, image work)
# =====================================================================
#
# Fixed/reviewed version of ``cpu_workloads.py``. The original is left
# unchanged for reference; this file corrects the logic bugs and makes the
# pool safe under the ``spawn`` start method (macOS/Windows default).
#
# Bugs fixed vs the original
# --------------------------
# 1. Deadlock: the original joined the worker processes *before* sending the
#    poison-pill sentinels, so the workers (which only exit on a sentinel)
#    never terminated. Here ``shutdown()`` sends one sentinel per worker first,
#    then joins.
# 2. Attribute access before validation: the original read ``payload.image_id``
#    before checking ``isinstance(...)`` (and again inside the "invalid"
#    branch), crashing on any non-``ImageProcessingRequest`` payload. Here the
#    type check happens before any attribute access.
# 3. Spawn safety: the original used ``target=self.process_task``, which pickles
#    the whole instance (including the queue) on spawn. Here the worker loop is
#    a module-level function and the queues are passed as explicit arguments,
#    which multiprocessing knows how to hand to children.
# 4. Results are now returned via a dedicated result queue, and the demo is
#    guarded by ``if __name__ == "__main__"`` (required for spawn).

from __future__ import annotations

import multiprocessing
import queue as queue_mod
from multiprocessing import JoinableQueue, Process, Queue

from pydantic import BaseModel


class ImageProcessingRequest(BaseModel):
    image_id: str
    image_data: bytes


class ProcessedImage(BaseModel):
    image_id: str
    size_bytes: int


def process_image(payload: ImageProcessingRequest) -> ProcessedImage:
    """Placeholder for the actual CPU-bound work (decode/resize/filter…).

    Kept as a small pure function so it can be unit-tested directly without
    spinning up any worker processes.
    """
    return ProcessedImage(image_id=payload.image_id, size_bytes=len(payload.image_data))


def _cpu_worker_loop(
    task_queue: JoinableQueue,
    result_queue: Queue | None,
    worker_id: int,
) -> None:
    """Consumer loop run in each worker process.

    Module-level (not a bound method) so it pickles cleanly under ``spawn``.
    Pulls payloads until it receives a ``None`` sentinel. Every ``get()`` is
    balanced by exactly one ``task_done()`` via the ``finally`` block, so
    ``JoinableQueue.join()`` in the parent unblocks correctly.
    """
    print(f"CPU worker {worker_id} started.")
    while True:
        payload = task_queue.get()
        try:
            if payload is None:  # poison pill -> shut this worker down
                break

            # Validate BEFORE touching any attributes (fixes the original crash).
            if not isinstance(payload, ImageProcessingRequest):
                print(f"CPU worker {worker_id}: skipping invalid payload {payload!r}")
                continue

            result = process_image(payload)
            if result_queue is not None:
                result_queue.put(result)
        finally:
            task_queue.task_done()  # balances every get(), incl. sentinel/invalid


class CpuWorkerQueue:
    """A pool of worker processes consuming CPU-bound tasks from a queue."""

    def __init__(self, num_workers: int | None = None, collect_results: bool = True) -> None:
        # Task queue (parent -> workers) and optional result queue (workers -> parent).
        self.task_queue: JoinableQueue = JoinableQueue()
        self.result_queue: Queue | None = Queue() if collect_results else None

        # Default to one worker per core; overridable for tests.
        self.num_workers = num_workers or multiprocessing.cpu_count()

        self.workers = [
            Process(
                target=_cpu_worker_loop,
                args=(self.task_queue, self.result_queue, i),
            )
            for i in range(self.num_workers)
        ]
        print(f"Starting {self.num_workers} CPU worker processes.")
        for worker in self.workers:
            worker.start()

    def insert_cpu_tasks(self, raw_payloads: list[ImageProcessingRequest]) -> None:
        """Producer entry point. Blocks if the queue's maxsize is reached."""
        for payload in raw_payloads:
            self.task_queue.put(payload)

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
        for worker in self.workers:
            worker.join()

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
    pool = CpuWorkerQueue(num_workers=4)

    payloads = [
        ImageProcessingRequest(image_id=f"image_{i}", image_data=b"fake_image_data" * (i + 1))
        for i in range(20)
    ]
    pool.insert_cpu_tasks(payloads)

    # Wait for all real work to finish, collect results, THEN shut the pool down.
    pool.join_tasks()
    results = pool.collect_results(len(payloads))
    pool.shutdown()

    print(f"Processed {len(results)} images.")
    for r in results[:3]:
        print(f"  {r.image_id}: {r.size_bytes} bytes")
