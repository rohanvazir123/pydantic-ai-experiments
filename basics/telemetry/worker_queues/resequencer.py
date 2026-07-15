# =====================================================================
# RESEQUENCING worker pool (asyncio): out-of-order telemetry -> seq order
# =====================================================================
#
# The other half of the worker-pool question: "ensuring out-of-order drone
# telemetry is resequenced correctly." The insight is that ordering breaks on
# the COMPLETION side, not arrival: with N concurrent workers and variable
# per-frame latency, worker A can *start* seq=5 before worker B *finishes*
# seq=4 and still finish first. Frames can arrive in perfect order and still
# come out scrambled.
#
# Same shape as io_workloads_fixed.py -- a Job on a queue, N Workers, a
# WorkerPool owning both (base.py) -- with the SQLAlchemy sink swapped for a
# Resequencer. The Resequencer is the only novel part; the pool is the same
# pool, and it stands in the same place TelemetryWriter does over there: the
# infrastructure the worker owns and hands to the job.
#
# Per device the Resequencer keeps only an `expected` counter, a min-heap of
# early arrivals, and the time its gap stalled. On each submit:
#
#   1. seq < expected -> already emitted, or skipped past; drop it.
#   2. otherwise push onto that device's min-heap.
#   3. emit the contiguous run from expected. Usually the whole story -- it
#      leaves the heap empty and nothing is withheld.
#   4. if a gap still holds frames back, give up on it once we have withheld
#      too many (max_buffer) or waited too long (max_delay) -- then drain the
#      heap in seq order, stepping over each gap and recording it.
#   5. close() at end of stream drains what's left, since (4) is only ever
#      reached from a submit() that will never come again.
#
# Why give up at all: a frame that is never coming would otherwise stall its
# device forever behind an ever-growing heap. Draining trades perfect ordering
# for liveness -- "fail-safe over fail-open". A heap that happens to be
# contiguous drains to zero gaps, so (3) and (4) differ only in how far they
# go, not in what they emit.
#
# Why TWO bounds -- they fail in opposite directions, so each covers the
# other's blind spot and both feed the same drain:
#   * max_buffer alone leaves ordering LATENCY unbounded. A device at 1 Hz sits
#     on a gap for a minute before filling a 60-frame buffer; a slow enough one
#     never fills it at all.
#   * max_delay alone leaves MEMORY unbounded, since a fast device can withhold
#     an enormous heap well inside the delay window.
#
# Scaling out: one Resequencer serialises every device through a single event
# loop, so it is a shard's throughput ceiling (though per frame it is only a
# heap push and a few pops -- the workers normally saturate first). Nothing
# couples one device to another; all state is keyed by device. So devices
# partition cleanly: hash device_id to a resequencer, a process, or a Kafka /
# Pulsar partition. That is the partition-key model those brokers are built on
# -- ordering holds only WITHIN a partition, and per-device ordering is all
# this needs.
#
# Known limits, named rather than built:
#   * No sweeper. Both bounds are only ever tested from submit(), so neither
#     fires for a device that goes quiet mid-gap; that tail waits for close().
#     Production ages out stalled gaps on a timer, which is what makes
#     max_delay a true latency bound rather than a best effort.
#   * A drain can outrun frames still in flight. It advances expected to the
#     largest buffered seq, but workers may still hold lower ones; those then
#     fail step 1 and are dropped as stale. Advancing only to heap[0][0] loses
#     less, at the cost of re-triggering once per gap.
#   * Per-device state is never reclaimed; a long-lived process wants the
#     device maps expired on disconnect.

from __future__ import annotations

import asyncio
import heapq
import random
import time
from itertools import count

from base import Job, JobStatus, Worker, WorkerPool


class TelemetryFrame(Job):
    """One telemetry sample from one device, ordered by ``seq`` per device."""

    device_id: str
    seq: int
    payload: dict[str, float]

    async def process(self, resequencer: Resequencer) -> None:
        """Do this job's work, then hand the finished frame to the resequencer.

        Mirrors ``TelemetryData.process(writer)``: the job does its own work and
        is passed the infrastructure it needs rather than holding it. The
        ``sleep`` stands in for real per-frame processing -- and being *variable*
        is the whole point, since that is what scrambles completion order.
        """
        await asyncio.sleep(random.uniform(0.001, 0.005))
        resequencer.submit(self)


class Resequencer:
    """Buffers out-of-order frames per device and emits them in seq order.

    The sink of this pool, in the position ``TelemetryWriter`` holds in the IO
    pool. Each device's stream is assumed to start at ``seq=0`` (a per-session
    counter, reset on reconnect). A gap is presumed lost once **either** bound
    is hit -- ``max_buffer`` (how many frames one device may withhold) or
    ``max_delay`` (how long it may withhold them); see the header for why both.
    """

    def __init__(self, max_buffer: int = 50, max_delay: float = 1.0) -> None:
        self.max_buffer = max_buffer
        self.max_delay = max_delay
        self._expected: dict[str, int] = {}  # device -> next seq to emit
        self._heaps: dict[str, list[tuple[int, int, TelemetryFrame]]] = {}  # device -> early arrivals
        self._blocked_since: dict[str, float] = {}  # device -> when its gap first stalled
        # Heap entries are (seq, tiebreak, frame). The tiebreak is the heapq
        # docs' recipe: without it, two frames with the same seq would compare
        # the *frames*, and pydantic models define no ordering -> TypeError.
        self._tiebreak = count()
        self.emitted: dict[str, list[TelemetryFrame]] = {}  # device -> frames, in seq order
        self.dropped: list[TelemetryFrame] = []
        self.gaps: list[tuple[str, int, int]] = []  # (device, skipped_from, skipped_to)

    def submit(self, frame: TelemetryFrame) -> None:
        """Buffer or emit ``frame``. Called once per completed frame.

        Needs no lock: every worker is a coroutine on one event loop, so the
        only concurrency is cooperative (at ``await`` points) and this method
        never awaits -- the same reasoning as ``IoWorkerPool._init_shared_state``.
        """
        device = frame.device_id
        expected = self._expected.setdefault(device, 0)
        heap = self._heaps.setdefault(device, [])

        # (1) Stale: this seq is already emitted, or was skipped by a gap.
        if frame.seq < expected:
            self.dropped.append(frame)
            return

        heapq.heappush(heap, (frame.seq, next(self._tiebreak), frame))  # (2)
        self._flush(device)

    def _flush(self, device: str, force: bool = False) -> None:
        """Emit as much as the ordering rules currently allow. (3, 4)

        Pops in seq order while the next frame is the one expected (3). On
        reaching a gap it stops and leaves the rest buffered -- unless a bound
        says that gap has held things up long enough, or ``force`` (how
        :meth:`close` releases the tail), in which case it steps over that gap
        and every later one, recording each (4).
        """
        heap = self._heaps[device]
        drain = force
        while heap:
            # (4) Blocked by a gap -- keep waiting, or give up on it for good?
            # The heap holds exactly the withheld frames at this point, so this
            # is the moment both bounds are meant to be judged on.
            if heap[0][0] > self._expected[device] and not drain:
                blocked_at = self._blocked_since.setdefault(device, time.monotonic())
                withheld_too_many = len(heap) > self.max_buffer
                waited_too_long = time.monotonic() - blocked_at > self.max_delay
                if not (withheld_too_many or waited_too_long):
                    return  # inside both bounds; the gap may still fill
                drain = True

            seq, _, frame = heapq.heappop(heap)
            if seq < self._expected[device]:
                # A duplicate buffered before its twin was emitted; too late now.
                self.dropped.append(frame)
                continue
            if seq > self._expected[device]:  # only reachable once draining
                self.gaps.append((device, self._expected[device], seq))
            self._expected[device] = seq + 1
            self.emitted.setdefault(device, []).append(frame)

        self._blocked_since.pop(device, None)  # heap empty -> nothing withheld

    def close(self) -> None:
        """End of stream: no frame can fill a gap now, so release everything. (5)

        Without this, whatever trails the last gap is withheld forever and never
        even surfaces as a gap -- (4) is only tested from :meth:`submit`, and
        there are no more submits coming.
        """
        for device in self._heaps:
            self._flush(device, force=True)


class ResequencingWorker(Worker):
    """One asyncio worker: pull frames until a sentinel, let each process itself.

    The worker holds the resequencer as an infrastructure resource and hands it
    to ``item.process(self.resequencer)`` -- it branches on nothing and knows
    nothing about heaps or sequence numbers, mirroring ``IoWorker`` calling
    ``item.process(self.writer)``.
    """

    def __init__(
        self,
        in_queue: asyncio.Queue,
        worker_id: int,
        resequencer: Resequencer,
        status: dict[str, JobStatus],
        cancelled: set[str],
    ) -> None:
        super().__init__(worker_id)
        self.in_queue = in_queue
        self.resequencer = resequencer
        self.status = status
        self.cancelled = cancelled

    async def run(self) -> None:
        while True:
            item = await self.in_queue.get()
            try:
                if item is None:  # poison pill
                    break
                if not isinstance(item, TelemetryFrame):
                    continue
                if item.job_id in self.cancelled:  # lazy cancel
                    # Note the knock-on: a cancelled frame never reaches the
                    # resequencer, so its seq becomes a gap like any lost packet
                    # and is drained past once a bound trips.
                    self.status[item.job_id] = JobStatus.CANCELLED
                    continue
                self.status[item.job_id] = JobStatus.RUNNING
                try:
                    # The worker does not know how to resequence; it just hands the frame to
                    # the resequencer and lets it do its own work. 
                    # The resequencer is a shared resource, so it is passed in by the pool.
                    # resequencer.submit() is called by the frame's process() method, 
                    # which is where the actual resequencing logic happens.
                    await item.process(self.resequencer)
                except Exception as exc:
                    # Contain the failure to THIS frame, exactly as IoWorker does:
                    # without this the error would bubble out of run(), kill the
                    # worker task, shrink the pool, and potentially hang
                    # in_queue.join(). `except Exception` (not bare `except`) lets
                    # asyncio.CancelledError -- a BaseException -- propagate so
                    # shutdown still cancels cleanly.
                    self.status[item.job_id] = JobStatus.FAILED
                    print(f"Resequencing worker {self.worker_id}: job {item.job_id} failed: {exc!r}")
                    continue
                self.status[item.job_id] = JobStatus.DONE
            except Exception as exc:
                # Safety net around the WHOLE loop body; the tight except above
                # only covers item.process(). Status bookkeeping must not kill
                # the worker task either.
                print(f"Resequencing worker {self.worker_id}: unexpected loop error: {exc!r}")
            finally:
                self.in_queue.task_done()


class ResequencingWorkerPool(WorkerPool):
    """Owns the asyncio queue, the Resequencer, and the pool of workers."""

    def __init__(
        self,
        resequencer: Resequencer | None = None,
        maxsize: int = 100,
        num_workers: int = 4,
    ) -> None:
        super().__init__(num_workers)
        #
        # Workflow this pool wires up  (parent = orchestrator):
        #
        #   parent.insert_job(frame)
        #        │
        #        ▼
        #   in_queue ───► worker[0..N] ───► frame.process(resequencer)
        #  (arrival order) (async tasks;            │
        #                   REORDER completion)     └──► Resequencer.submit()
        #                                                  │
        #                                                  └──► emitted[device]
        #                                                        (strict seq order)
        #
        # No result queue (as with the IO pool): a frame's "result" is its place
        # in the emitted sequence, so the Resequencer IS the sink.
        #
        # __init__ reads as ordered steps; each helper owns one concern. Must run
        # inside an event loop -- _start_workers uses asyncio.create_task.
        self._init_queue(maxsize)
        self._init_shared_state()
        self._init_resequencer(resequencer)
        self._start_workers()

    def _init_queue(self, maxsize: int) -> None:
        """The single asyncio queue: parent -> workers, with backpressure at
        ``maxsize`` (``insert_job`` awaits when the queue is full)."""
        self.in_queue: asyncio.Queue = asyncio.Queue(maxsize)

    def _init_shared_state(self) -> None:
        """Job bookkeeping. As in ``IoWorkerPool``, no ``Manager`` is needed:
        asyncio runs one process on one event-loop thread, so a plain dict/set
        is safe.

        * ``_status``    job_id -> JobStatus  (workers write, parent reads on GET)
        * ``_cancelled`` set of flagged job_ids (parent adds, workers check)
        """
        self._status: dict[str, JobStatus] = {}
        self._cancelled: set[str] = set()

    def _init_resequencer(self, resequencer: Resequencer | None) -> None:
        """The sink shared by all workers. Sized by the caller: see the header on
        picking ``max_buffer`` / ``max_delay``."""
        self.resequencer = resequencer or Resequencer()

    def _start_workers(self) -> None:
        """Build one ``ResequencingWorker`` per slot and launch each as a task.

        Every worker shares the SAME queue, resequencer, and bookkeeping -- only
        the worker id differs. Requires a running event loop.
        """
        self.reseq_workers = [
            ResequencingWorker(self.in_queue, i, self.resequencer, self._status, self._cancelled)
            for i in range(self.num_workers)
        ]
        self.workers = [asyncio.create_task(w.run()) for w in self.reseq_workers]

    # -- Public API ---------------------------------------------------------
    # Same shape as IoWorkerPool, so the same REST mapping applies: `POST /frames`
    # -> insert_job(); `GET /jobs/{id}` -> get_job_status(); `DELETE /jobs/{id}`
    # -> cancel_job(). The ordered stream is read off resequencer.emitted.

    async def insert_job(self, job: TelemetryFrame) -> None:
        """Submit one frame and mark it ``QUEUED``; await if the queue is full."""
        self._status[job.job_id] = JobStatus.QUEUED
        await self.in_queue.put(job)

    def cancel_job(self, job_id: str) -> bool:
        """Lazily cancel a still-queued frame (a worker skips it when reached)."""
        if self._status.get(job_id, JobStatus.UNKNOWN) != JobStatus.QUEUED:
            return False
        self._cancelled.add(job_id)
        self._status[job_id] = JobStatus.CANCELLED
        return True

    def get_job_status(self, job_id: str) -> JobStatus:
        """Current status of a frame, or ``UNKNOWN`` if this pool never saw it."""
        return self._status.get(job_id, JobStatus.UNKNOWN)

    async def join_tasks(self) -> None:
        """Block until every submitted frame has been processed.

        The result barrier: after this returns, every frame has reached the
        resequencer -- though frames may still be *withheld* behind a gap until
        :meth:`shutdown` closes the stream.
        """
        await self.in_queue.join()

    # collect_results() is inherited from WorkerPool: a frame's "result" is
    # whether it was processed (DONE) -- reported from _status. The ordered
    # output itself lives in resequencer.emitted, the sink.

    async def shutdown(self) -> None:
        """Sentinel each worker, await them, then close the stream.

        ``close()`` must come last: it presumes no frame can fill a gap any
        more, which is only true once no worker is still holding one.
        """
        for _ in range(self.num_workers):
            await self.in_queue.put(None)
        await asyncio.gather(*self.workers)
        self.resequencer.close()


async def main() -> None:

    # Create a pool of resequencing workers and submit 30 frames IN seq order --
    # the pool itself is what scrambles them. seq=15 is never produced, standing
    # in for a packet lost for good. max_buffer sits above the reorder distance
    # ~num_workers concurrent frames introduce (so ordinary jitter never trips
    # it) and below the 14 frames trailing seq=15 (so that gap is given up on
    # mid-stream rather than waiting for close()).
    pool = ResequencingWorkerPool(num_workers=4, resequencer=Resequencer(max_buffer=10))
    for i in range(30):
        if i == 15:
            continue
        await pool.insert_job(
            TelemetryFrame(device_id="drone-1", seq=i, payload={"alt_m": float(i)})
        )

    # Wait for all frames to finish (to reach the resequencer) before shutdown.
    await pool.join_tasks()

    # Shutdown the pool cleanly (sentinels + await workers, then close the stream)
    await pool.shutdown()

    # Check how many frames ran successfully, and what order they came out in
    outcomes = pool.collect_results()
    done = sum(1 for status in outcomes.values() if status == JobStatus.DONE)
    print(f"{done}/{len(outcomes)} frames processed; gaps: {pool.resequencer.gaps}")
    print("emitted order:", [f.seq for f in pool.resequencer.emitted["drone-1"]])


if __name__ == "__main__":
    asyncio.run(main())
