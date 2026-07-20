"""Moving average of a streaming telemetry metric over the last N seconds.
"""

from __future__ import annotations

import math
import time
from collections import deque
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

    Not handled: delayed samples. The window is just "now - max_age", so a
    sample that arrives after its timestamp has already aged out is added and
    evicted in the same call -- it never contributes. There's no watermark or
    grace period holding the window open for stragglers.
    """

    def __init__(
        self,
        max_age_seconds: float,
        *,
        clock: Callable[[], float] | None = None,
        ) -> None:
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        self._max_age = max_age_seconds
        self._buffer: deque[Sample] = deque()  # kept sorted ascending by timestamp
        self._running_sum = 0.0

        # The clock is injectable for deterministic testing; default is monotonic.
        self._clock: Callable[[], float] = clock or time.monotonic  # default monotonic clock

    def add_batch(
        self, sorted_batch: Iterable[Sample], *, now: float | None = None
    ) -> None:
        """Add a batch of samples, already sorted ascending by timestamp.

        Precondition: ``sorted_batch`` is sorted; this is trusted, not
        re-verified, for throughput. ``now`` may be supplied to make eviction
        deterministic; otherwise the clock is read. An empty batch is a valid
        "tick" that just refreshes the window.
        """
        batch = list(sorted_batch)
        if not batch:
            self._evict(self._clock() if now is None else now)
            return

        self._running_sum += math.fsum(s.value for s in batch)
        prev_last = self._buffer[-1].timestamp if self._buffer else None
        self._buffer.extend(batch)

        if prev_last is not None and batch[0].timestamp < prev_last:
            # batch overlaps existing data -- re-sort, O(N + k). deque has no
            # in-place sort, so rebuild it from a sorted list.
            self._buffer = deque(sorted(self._buffer))

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
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._running_sum -= self._buffer.popleft().value


def _demo() -> None:
    """Feed a burst of readings and show the average shrinking as they age out."""
    clock_time = [1000.0]
    # Passing a lambda clock allows us to control the time for testing.
    avg = TelemetryRollingAverage(max_age_seconds=5.0, clock=lambda: clock_time[0])

    for i in range(10):
        avg.add_batch([Sample(timestamp=1000.0 + i, value=float(i))])
        clock_time[0] = 1000.0 + i
        print(
            f"t={clock_time[0]:.0f}  count={avg.window_count():2d}  "
            f"avg={avg.get_moving_average():.2f}"
        )

    clock_time[0] += 100  # let everything age out
    print(f"after quiet period: count={avg.window_count()} avg={avg.get_moving_average()}")


if __name__ == "__main__":
    _demo()
