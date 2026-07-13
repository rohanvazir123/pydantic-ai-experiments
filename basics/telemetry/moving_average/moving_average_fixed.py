"""Moving average of a streaming telemetry metric over the last N seconds.

Reviewed, corrected version of the ``basics/telemetry/rolling_avg.py`` draft.
The draft is kept unchanged for comparison; build on this file instead.

What changed (see README for the full write-up):

1. **Read was stale with no new data.** The draft only evicted expired points
   inside ``add_batch``. If a metric went quiet, ``get_moving_average`` kept
   counting points that had already aged out of the window. Here every read
   (``get_moving_average`` / ``window_count``) evicts first, so the "last N
   seconds" contract holds even when nothing is arriving.

2. **Wall clock → monotonic, and injectable.** Windowing on ``time.time()``
   breaks if the system clock steps backwards (NTP). The default clock is now
   ``time.monotonic``; tests inject a fake clock (or an explicit ``now``) for
   determinism.

3. **O(N log N) full re-sort → O(N + k), only when needed.** Arrivals can be out
   of order, so the buffer must stay sorted. The draft re-sorted the whole buffer
   on *every* batch. Here the common in-order case is a pure O(k) ``extend``, and
   only a batch that overlaps existing data triggers ``list.sort()`` — Timsort is
   adaptive, so merging the two already-sorted runs is O(N + k) (and in C, in
   place). (A ``SortedList`` would give O(log N) per-insert without a full
   rescan; we stay stdlib-only — see the README's data-structure evaluation.)

4. **One source of truth for the count.** The draft tracked ``running_count``
   separately from the buffer, so the two could drift. Count is now just
   ``len(buffer)``; only the sum is memoised (for O(1) reads).

5. **Single-owner concurrency — no internal lock, by design.** This is built for
   one owner: a single thread, or (typically) a single ``asyncio`` loop, where
   the synchronous methods are atomic because nothing preempts them mid-call.
   There is deliberately **no** ``threading.Lock`` — mixing one with asyncio
   invites deadlocks (a lock held across an ``await`` freezes the one loop
   thread), and it isn't needed under a single loop. If you genuinely share the
   instance across OS threads, guard it externally; if a future critical section
   spans ``await``s, use ``asyncio.Lock``. Note the GIL does **not** substitute:
   it protects CPython internals, not compound ops like ``running_sum += x``.
   ``resync()`` re-derives the memoised sum with ``math.fsum`` to clear
   floating-point drift from long add/subtract runs.

6. **Fully typed.** Samples are a ``NamedTuple`` instead of loose ``dict``s.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from typing import NamedTuple


class Sample(NamedTuple):
    """One telemetry reading: an epoch/monotonic timestamp and its value."""

    timestamp: float
    value: float


class TelemetryRollingAverage:
    """Mean of all samples whose timestamp falls within the last N seconds.

    Reads are O(1) (the window sum is memoised); ``add_batch`` is O(k) on the
    in-order fast path and O(N + k) when a batch interleaves with existing data.
    Eviction is amortised O(1) per sample.

    Concurrency: single-owner — one thread, or one asyncio loop (the methods are
    synchronous with no ``await`` inside, so a single loop serialises them for
    free). Not internally locked; see the module docstring for the contract and
    when to reach for ``asyncio.Lock`` / external guarding.
    """

    def __init__(
        self,
        max_age_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        self._max_age = max_age_seconds
        self._clock = clock
        self._buffer: list[Sample] = []  # kept sorted ascending by timestamp
        self._running_sum = 0.0

    def add(self, *samples: Sample, now: float | None = None) -> None:
        """Add one or more samples (convenience wrapper over :meth:`add_batch`).

        ``*samples`` captures the positional args, which also keeps ``now``
        keyword-only. Passing more than one at a time carries
        :meth:`add_batch`'s precondition — they must be in ascending timestamp
        order. Calling with no samples is a valid "tick" that just refreshes the
        window.
        """
        self.add_batch(samples, now=now)

    def add_batch(
        self, sorted_batch: Iterable[Sample], *, now: float | None = None
    ) -> None:
        """Add a batch of samples that is already sorted ascending by timestamp.

        Precondition: ``sorted_batch`` is internally sorted. This is trusted (not
        re-verified) for throughput, matching the parameter name. ``now`` may be
        supplied to make eviction deterministic; otherwise the clock is read.
        """
        batch = list(sorted_batch)
        if not batch:
            # An empty poke still refreshes the window, so a periodic tick with
            # no data keeps the average honest.
            self._evict(self._clock() if now is None else now)
            return

        self._running_sum += math.fsum(s.value for s in batch)
        prev_last = self._buffer[-1].timestamp if self._buffer else None
        self._buffer.extend(batch)
        if prev_last is not None and batch[0].timestamp < prev_last:
            # The batch overlaps existing data → buffer is now two sorted runs.
            # Timsort is adaptive: it detects the runs and merges them in O(N + k)
            # (in C, in place). Sample is a NamedTuple, so it already orders by
            # timestamp first — no key needed. In-order batches skip the sort and
            # stay a pure O(k) append.
            self._buffer.sort()
        self._evict(self._clock() if now is None else now)

    def get_moving_average(self, *, now: float | None = None) -> float:
        """Current mean over the window, or ``0.0`` when the window is empty.

        Evicts expired samples first, so the result is correct even if no new
        data has arrived since the last call.
        """
        self._evict(self._clock() if now is None else now)
        if not self._buffer:
            return 0.0
        return self._running_sum / len(self._buffer)

    def window_count(self, *, now: float | None = None) -> int:
        """Number of samples currently inside the window (evicts first)."""
        self._evict(self._clock() if now is None else now)
        return len(self._buffer)

    def resync(self) -> None:
        """Re-derive the memoised sum from the buffer with ``math.fsum``.

        Incrementally adding and subtracting floats accumulates rounding error
        over long runs. Call this periodically to reset that drift exactly.
        """
        self._running_sum = math.fsum(s.value for s in self._buffer)

    def _evict(self, now: float) -> None:
        """Drop samples older than the window (oldest-first from the front)."""
        cutoff = now - self._max_age
        expired = 0
        expired_sum = 0.0
        for sample in self._buffer:  # buffer is sorted → oldest first
            if sample.timestamp < cutoff:
                expired_sum += sample.value
                expired += 1
            else:
                break
        if expired:
            self._running_sum -= expired_sum
            del self._buffer[:expired]


def _demo() -> None:
    """Feed a burst of readings and show the average shrinking as they age out."""
    clock_time = [1000.0]
    avg = TelemetryRollingAverage(max_age_seconds=5.0, clock=lambda: clock_time[0])

    for i in range(10):
        avg.add(Sample(timestamp=1000.0 + i, value=float(i)))
        clock_time[0] = 1000.0 + i
        print(
            f"t={clock_time[0]:.0f}  count={avg.window_count():2d}  "
            f"avg={avg.get_moving_average():.2f}"
        )

    clock_time[0] += 100  # let everything age out
    print(f"after quiet period: count={avg.window_count()} avg={avg.get_moving_average()}")


if __name__ == "__main__":
    _demo()
