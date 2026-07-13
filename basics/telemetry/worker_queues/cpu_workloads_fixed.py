# =====================================================================
# CPU-BOUND worker pool (multiprocessing: parsing, aggregating, image work)
# =====================================================================
#
# Fixed/reviewed version of ``cpu_workloads.py``. The original is left
# unchanged for reference; this file corrects the logic bugs and makes the
# pool safe under the ``spawn`` start method (macOS/Windows default).
#
# Responsibilities are split into three classes:
#   * ``ImageProcessor``  — the actual CPU-bound work (stateless, unit-testable)
#   * ``CpuWorker``       — one worker process's consume loop
#   * ``CpuWorkerPool``  — owns the queues and the pool of worker processes
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
#    instance whose only state is the (spawn-picklable) queues, an int id, and
#    the processor — it never references the pool manager.

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


class ImageProcessor:
    """The CPU-bound work itself, isolated from any queue/transport concerns.

    Stateless and picklable, so it can be handed to worker processes and can be
    unit-tested directly without spinning up the pool. Swap in a subclass to do
    real decoding/resizing/filtering.
    """

    def process(self, payload: ImageProcessingRequest) -> ProcessedImage:
        # Placeholder for real image work (decode/resize/filter…).
        return ProcessedImage(image_id=payload.image_id, size_bytes=len(payload.image_data))


class CpuWorker:
    """One worker process's consume loop.

    Holds only spawn-picklable state (the queues, an int id, and the
    processor), so its :meth:`run` method can be used directly as a
    ``Process`` target under the ``spawn`` start method.
    """

    def __init__(
        self,
        task_queue: JoinableQueue,
        result_queue: Queue | None,
        worker_id: int,
        processor: ImageProcessor,
    ) -> None:
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.processor = processor

    def run(self) -> None:
        """Pull payloads until a ``None`` sentinel; balance every get with a done."""
        print(f"CPU worker {self.worker_id} started.")
        while True:
            payload = self.task_queue.get()
            try:
                if payload is None:  # poison pill -> shut this worker down
                    break

                # Validate BEFORE touching any attributes (fixes the original crash).
                if not isinstance(payload, ImageProcessingRequest):
                    print(f"CPU worker {self.worker_id}: skipping invalid {payload!r}")
                    continue

                result = self.processor.process(payload)
                if self.result_queue is not None:
                    self.result_queue.put(result)
            finally:
                self.task_queue.task_done()  # balances every get(), incl. sentinel/invalid


class CpuWorkerPool:
    """Owns the task/result queues and the pool of :class:`CpuWorker` processes."""

    def __init__(
        self,
        num_workers: int | None = None,
        collect_results: bool = True,
        processor: ImageProcessor | None = None,
    ) -> None:
        # Task queue (parent -> workers) and optional result queue (workers -> parent).
        self.task_queue: JoinableQueue = JoinableQueue()
        self.result_queue: Queue | None = Queue() if collect_results else None

        # Default to one worker per core; overridable for tests.
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.processor = processor or ImageProcessor()

        # Each worker is its own object; only worker state crosses to the child.
        self.workers = [
            CpuWorker(self.task_queue, self.result_queue, i, self.processor)
            for i in range(self.num_workers)
        ]
        self.processes = [Process(target=worker.run) for worker in self.workers]

        print(f"Starting {self.num_workers} CPU worker processes.")
        for process in self.processes:
            process.start()

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
        for process in self.processes:
            process.join()

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
