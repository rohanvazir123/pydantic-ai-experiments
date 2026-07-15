# =====================================================================
# RESEQUENCING worker pool (asyncio): out-of-order telemetry -> seq order
# =====================================================================
#
# The other half of the worker-pool question: "ensuring out-of-order drone
# telemetry is resequenced correctly." Ordering breaks on the COMPLETION side,
# not arrival: with N workers and variable per-frame latency, A can start seq=5
# before B finishes seq=4 and still finish first.
#
# Same shape as io_workloads_fixed.py (Job -> Worker -> WorkerPool from
# base.py), with the SQLAlchemy sink swapped for a Resequencer -- which sits
# exactly where TelemetryWriter sits: infrastructure the worker owns and hands
# to the job.
#
# Per device: an `expected` counter, a min-heap of early arrivals, and when its
# gap stalled. On each submit:
#
#   1. seq < expected -> already emitted or skipped past; drop it.
#   2. else push onto that device's min-heap.
#   3. emit the contiguous run from expected -- often NOTHING. Withheld frames
#      just stay put; a LATER submit releases them (the frame that fills the gap
#      triggers the emit that drains everything behind it).
#   4. only if a gap still blocks: give up once we have withheld too many
#      (max_buffer) or waited too long (max_delay), then empty the heap in seq
#      order, recording each gap stepped over.
#   5. close() releases the tail at end of stream, since (4) only ever runs from
#      a submit() that will never come.
#
# Concurrency: ONE thread, ONE core, N coroutines. Exactly one worker executes
# Python at any instant -- which is WHY the heap needs no lock (submit() has no
# await, so nothing interleaves with it). The speedup is N I/O ops in flight
# while their workers sit parked: the waits overlap, not the code. Measured,
# 8 workers x 50ms writes -> 7.8x (peak 1 executing / 8 waiting); the same pool
# over CPU-bound work -> 1.1x, since computing never yields.
#
# Nothing ever waits on a frame. A blocked device is data in a heap, not a
# parked coroutine: (3) returns in ~2us and the worker goes straight back for
# the next frame. Even "waited too long" is a timestamp compared on a later
# submit, not a wait.
#
# Why give up at all: a frame that is never coming would stall its device
# forever behind a growing heap. Draining trades ordering for liveness.
#
# Why TWO bounds: max_buffer alone leaves LATENCY unbounded (a 1 Hz device sits
# on a gap for a minute before filling a 60-frame buffer); max_delay alone
# leaves MEMORY unbounded (a fast device withholds a huge heap inside the
# window). Note max_delay only half-delivers without a sweeper -- see limits.
#
# Scaling out: one Resequencer serialises every device on one loop, so it is a
# shard's ceiling -- but nothing couples devices (all state is keyed by device),
# so hash device_id to a resequencer / process / Kafka partition. Ordering holds
# only WITHIN a partition, which is all per-device ordering needs.
#
# Taken further: PARTITION AT INGRESS, one worker owning a device for life.
# This design takes the opposite side, so the trade-offs are worth naming:
#   + No shared state at all -- the lock question stops existing rather than
#     being answered, and submit() could then be async (i.e. a real DB sink).
#   + One owner processing serially cannot reorder its own device, so
#     completion reordering disappears. This is the Kafka / Flink model.
#   - Per-device parallelism drops to 1; a hot drone cannot be spread. This
#     design buys that parallelism and pays for it with the shared buffer.
#   - Hashing balances device COUNT, not load -- one chatty drone makes its
#     worker hot (Kafka's hot partition; fix is many more slots than workers).
#   - Head-of-line blocking moves to ingress: N queues, and one full queue
#     blocks the producer for every other device. Wants a drop policy, not a wait.
#   - Ownership must move when N changes or devices come and go (consistent
#     hashing / consumer-group rebalance).
#   - It does NOT remove this class: arrival can still be out of order (UDP), so
#     each worker keeps its own per-device buffer -- same algorithm, unshared.
# Sizing: a parked coroutine costs ~KB, so one worker PER DEVICE is viable into
# the thousands before hashing onto a fixed worker count is worth it.
#
# Known limits, named not built:
#   * No sweeper -- both bounds are only ever READ from submit(), so a timed-out
#     heap does not emit until that device's NEXT frame arrives. max_delay is
#     therefore not a latency bound at all: effective release is max_delay PLUS
#     the inter-frame interval (~1s for a 1 Hz drone), and for a device that
#     goes quiet mid-gap, never -- close() is the only thing that frees it,
#     i.e. it fails exactly when something has gone wrong. Fix is ~10 lines: a
#     task that periodically calls _emit_ready() for every device, which
#     re-checks waited_too_long. Stays lock-free -- _emit_ready is sync, so a
#     sweeper cannot interleave with a worker mid-emit.
#   * A drain can outrun in-flight frames -- it advances expected to the largest
#     buffered seq while a worker may still hold a lower one, which then fails
#     step 1 and is dropped despite arriving intact.
#   * Per-device state is never reclaimed; expire it on disconnect.

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
        """Do this job's work, then hand the frame to the resequencer.

        Mirrors ``TelemetryData.process(writer)``: the job does its own work and
        is passed the infrastructure it needs rather than holding it.
        """
        # SIMULATED I/O-bound work -- a lookup, an inference call. NOT the DB
        # write: that is a sink and belongs at emit (sketched below).
        #
        # This await is where concurrency kicks in -- the only real suspension a
        # worker has, since queue.get()/put() return inline unless empty/full.
        # Worker 1 parks here, worker 2 takes the core and parks here too: N
        # waits overlap on one thread. Delete it and worker 0 drains alone.
        #
        # BEFORE submit on purpose -- that is what scrambles completion order.
        # After submit, submits arrive in dequeue order, already sorted.
        await asyncio.sleep(random.uniform(0.001, 0.005))
        resequencer.submit(self)

        # A real DB sink goes at EMIT, not here, and persists what the
        # resequencer RELEASED -- not `self`, which may still be buffered:
        #
        #     resequencer.submit(self)              # sync, atomic: decides order
        #     for f in resequencer.take_emitted():  # whatever that released
        #         await writer.write(f)             # SQLAlchemy, in seq order
        #
        # submit() must stay sync: an await inside it breaks the atomicity that
        # stands in for a lock. It would also move the disorder to arrival, so
        # main() would shuffle its input to model UDP reordering -- the real
        # cause in production.


class Resequencer:
    """Buffers out-of-order frames per device and emits them in seq order.

    This pool's sink, where the IO pool has ``TelemetryWriter``. Each device's
    stream is assumed to start at ``seq=0`` (a per-session counter, reset on
    reconnect). A gap is presumed lost once **either** bound is hit --
    ``max_buffer`` (frames withheld) or ``max_delay`` (time withheld); the
    header covers why both.
    """

    def __init__(self, max_buffer: int = 50, max_delay: float = 1.0) -> None:
        self.max_buffer = max_buffer
        self.max_delay = max_delay
        self._expected: dict[str, int] = {}  # device -> next seq to emit
        self._heaps: dict[str, list[tuple[int, int, TelemetryFrame]]] = {}  # device -> early arrivals
        self._blocked_since: dict[str, float] = {}  # device -> when its gap stalled
        # Entries are (seq, tiebreak, frame). Without the tiebreak (the heapq
        # docs' recipe) two frames of equal seq would compare the *frames*, and
        # pydantic models define no ordering -> TypeError.
        self._tiebreak = count()
        self.emitted: dict[str, list[TelemetryFrame]] = {}  # device -> frames, in seq order
        self.dropped: list[TelemetryFrame] = []
        self.gaps: list[tuple[str, int, int]] = []  # (device, skipped_from, skipped_to)

    def submit(self, frame: TelemetryFrame) -> None:
        """Buffer or emit ``frame``. Called once per completed frame.

        Synchronous on purpose: with no ``await``, a worker runs this to
        completion and nothing can interleave, so the read-modify-write across
        ``_expected[device]`` and the heap is atomic without a lock. Add an
        await here and that breaks silently -- no test drives two coroutines
        through one device at once.
        """
        device = frame.device_id
        expected = self._expected.setdefault(device, 0)
        heap = self._heaps.setdefault(device, [])

        # (1) Stale: already emitted, or skipped by a gap.
        if frame.seq < expected:
            self.dropped.append(frame)
            return

        heapq.heappush(heap, (frame.seq, next(self._tiebreak), frame))  # (2)
        self._emit_ready(device)

    def _emit_ready(self, device: str, force: bool = False) -> None:
        """Emit what ordering permits -- routinely nothing.

        **Not a drain.** The usual outcome is the early ``return`` at the gap
        check, leaving the heap untouched. It only empties when the next frame
        IS the expected one (3), or a bound expired -- or ``force``, for
        :meth:`close` -- and we step over the gap for good (4). Blocked frames
        are released by a LATER submit, not by this call looping until they are.
        """
        heap = self._heaps[device]
        drain = force
        while heap:  # a guard -- "is anything releasable?" -- NOT "until empty"
            blocked_by_gap = heap[0][0] > self._expected[device]

            # ---- THE EARLY EXIT (4) -- the only way out but an empty heap.
            # Past here every path pops, so the heap shrinks each iteration and
            # the loop cannot spin.
            if blocked_by_gap and not drain:
                blocked_at = self._blocked_since.setdefault(device, time.monotonic())
                withheld_too_many = len(heap) > self.max_buffer
                waited_too_long = time.monotonic() - blocked_at > self.max_delay
                if not (withheld_too_many or waited_too_long):
                    return  # heap untouched; the missing frame may yet arrive
                drain = True  # gave up: step over this gap and every later one
            # --------------------------------------------------------------

            seq, _, frame = heapq.heappop(heap)
            if seq < self._expected[device]:
                self.dropped.append(frame)  # dup buffered before its twin emitted
                continue
            if seq > self._expected[device]:  # only reachable once draining
                self.gaps.append((device, self._expected[device], seq))
            self._expected[device] = seq + 1
            self.emitted.setdefault(device, []).append(frame)

        self._blocked_since.pop(device, None)  # heap empty -> nothing withheld

    def close(self) -> None:
        """End of stream: no frame can fill a gap now, so release everything. (5)

        Without this the tail behind the last gap is withheld forever and never
        even surfaces as a gap -- (4) only runs from :meth:`submit`, and there
        are no more coming.
        """
        for device in self._heaps:
            self._emit_ready(device, force=True)


class ResequencingWorker(Worker):
    """One asyncio worker: pull frames until a sentinel, let each process itself.

    Holds the resequencer as an infrastructure resource and hands it to
    ``item.process(...)`` -- it branches on nothing and knows nothing about
    heaps or sequence numbers, mirroring ``IoWorker`` passing its writer.
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
                    # Knock-on: a cancelled frame never reaches the resequencer,
                    # so its seq reads as a gap like any lost packet.
                    self.status[item.job_id] = JobStatus.CANCELLED
                    continue
                self.status[item.job_id] = JobStatus.RUNNING
                try:
                    # The worker knows nothing about ordering: it hands the job
                    # the resequencer and the job does its own work.
                    await item.process(self.resequencer)
                except Exception as exc:
                    # Contain the failure to THIS frame, as IoWorker does: else it
                    # bubbles out of run(), kills the task, shrinks the pool and can
                    # hang in_queue.join(). `except Exception` lets CancelledError
                    # (a BaseException) through so shutdown still works.
                    self.status[item.job_id] = JobStatus.FAILED
                    print(f"Resequencing worker {self.worker_id}: job {item.job_id} failed: {exc!r}")
                    continue
                self.status[item.job_id] = JobStatus.DONE
            except Exception as exc:
                # Safety net for the rest of the loop body; the except above only
                # covers process(). Bookkeeping must not kill the task either.
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
        # Ordered steps, one concern each. Must run inside an event loop --
        # _start_workers uses asyncio.create_task.
        self._init_queue(maxsize)
        self._init_shared_state()
        self._init_resequencer(resequencer)
        self._start_workers()

    def _init_queue(self, maxsize: int) -> None:
        """Parent -> workers, with backpressure at ``maxsize``."""
        self.in_queue: asyncio.Queue = asyncio.Queue(maxsize)

    def _init_shared_state(self) -> None:
        """Job bookkeeping. As in ``IoWorkerPool``, no ``Manager`` needed: one
        process on one event-loop thread, so a plain dict/set is safe.

        * ``_status``    job_id -> JobStatus  (workers write, parent reads)
        * ``_cancelled`` flagged job_ids (parent adds, workers check)
        """
        self._status: dict[str, JobStatus] = {}
        self._cancelled: set[str] = set()

    def _init_resequencer(self, resequencer: Resequencer | None) -> None:
        """The sink shared by all workers; the caller sizes its bounds."""
        self.resequencer = resequencer or Resequencer()

    def _start_workers(self) -> None:
        """One worker per slot, each an asyncio task.

        All share the SAME queue, resequencer and bookkeeping -- only the id
        differs. Requires a running event loop.
        """
        self.reseq_workers = [
            ResequencingWorker(self.in_queue, i, self.resequencer, self._status, self._cancelled)
            for i in range(self.num_workers)
        ]
        self.workers = [asyncio.create_task(w.run()) for w in self.reseq_workers]

    # -- Public API ---------------------------------------------------------
    # Same shape as IoWorkerPool, so the same REST mapping: `POST /frames` ->
    # insert_job(); `GET /jobs/{id}` -> get_job_status(); `DELETE /jobs/{id}` ->
    # cancel_job(). The ordered stream is read off resequencer.emitted.

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
        """Current status of a frame, or ``UNKNOWN`` if never seen."""
        return self._status.get(job_id, JobStatus.UNKNOWN)

    async def join_tasks(self) -> None:
        """Block until every submitted frame has reached the resequencer.

        Frames may still be *withheld* behind a gap until :meth:`shutdown`
        closes the stream.
        """
        await self.in_queue.join()

    # collect_results() is inherited from WorkerPool: a frame's "result" is
    # whether it was processed (DONE). The ordered output lives in
    # resequencer.emitted, the sink.

    async def shutdown(self) -> None:
        """Sentinel each worker, await them, then close the stream.

        ``close()`` comes last: it presumes no frame can fill a gap any more,
        which only holds once no worker is still holding one.
        """
        for _ in range(self.num_workers):
            await self.in_queue.put(None)
        await asyncio.gather(*self.workers)
        self.resequencer.close()


async def main() -> None:

    # Create a pool and submit 30 frames IN seq order -- the pool is what
    # scrambles them. seq=15 is never produced: a packet lost for good.
    # max_buffer sits above the reorder distance ~4 concurrent frames cause (so
    # jitter never trips it) and below the 14 frames trailing seq=15 (so that
    # gap is given up on mid-stream rather than at close()).
    pool = ResequencingWorkerPool(num_workers=4, resequencer=Resequencer(max_buffer=10))
    for i in range(30):
        if i == 15:
            continue
        await pool.insert_job(
            TelemetryFrame(device_id="drone-1", seq=i, payload={"alt_m": float(i)})
        )

    # Wait for all frames to reach the resequencer before shutdown.
    await pool.join_tasks()

    # Shutdown cleanly (sentinels + await workers, then close the stream)
    await pool.shutdown()

    # Check how many ran, and what order they came out in
    outcomes = pool.collect_results()
    done = sum(1 for status in outcomes.values() if status == JobStatus.DONE)
    print(f"{done}/{len(outcomes)} frames processed; gaps: {pool.resequencer.gaps}")
    print("emitted order:", [f.seq for f in pool.resequencer.emitted["drone-1"]])


if __name__ == "__main__":
    asyncio.run(main())
