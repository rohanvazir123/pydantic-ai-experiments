"""Tests for the corrected moving-average implementation.

Run from the repo root with the project venv:

    .venv/bin/python -m pytest basics/telemetry/moving_average/test_moving_average_fixed.py -v

Time is driven by an injected fake clock (or an explicit ``now``), so eviction
behavior is deterministic with no ``sleep``.
"""

import asyncio
import math

import pytest
from moving_average import Sample, TelemetryRollingAverage


class FakeClock:
    """A hand-cranked clock so window/eviction behavior is deterministic."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_basic_average() -> None:
    clock = FakeClock(100.0)
    avg = TelemetryRollingAverage(max_age_seconds=10.0, clock=clock)
    avg.add_batch([Sample(100.0, 2.0), Sample(100.0, 4.0)])
    assert avg.get_moving_average() == 3.0
    assert avg.window_count() == 2


def test_empty_window_returns_zero() -> None:
    avg = TelemetryRollingAverage(max_age_seconds=10.0, clock=FakeClock())
    assert avg.get_moving_average() == 0.0
    assert avg.window_count() == 0


def test_read_evicts_with_no_new_data() -> None:
    """The key fix: a quiet metric must not keep counting aged-out samples."""
    clock = FakeClock(100.0)
    avg = TelemetryRollingAverage(max_age_seconds=10.0, clock=clock)
    avg.add_batch([Sample(100.0, 2.0), Sample(101.0, 4.0)])
    assert avg.get_moving_average() == 3.0

    clock.advance(100.0)  # now=200, cutoff=190 → both samples aged out
    assert avg.get_moving_average() == 0.0
    assert avg.window_count() == 0


def test_partial_eviction() -> None:
    clock = FakeClock(100.0)
    avg = TelemetryRollingAverage(max_age_seconds=10.0, clock=clock)
    avg.add_batch([Sample(100.0, 10.0), Sample(108.0, 20.0)])
    clock.advance(9.0)  # now=109, cutoff=99 → 100 stays; advance more below
    assert avg.window_count() == 2
    clock.advance(2.0)  # now=111, cutoff=101 → sample@100 evicted, @108 stays
    assert avg.window_count() == 1
    assert avg.get_moving_average() == 20.0


def test_out_of_order_batch_is_merged() -> None:
    clock = FakeClock(102.0)
    avg = TelemetryRollingAverage(max_age_seconds=100.0, clock=clock)
    avg.add_batch([Sample(100.0, 1.0), Sample(102.0, 3.0)])
    avg.add_batch([Sample(101.0, 5.0)])  # older than tail → merge path

    assert [s.timestamp for s in avg._buffer] == [100.0, 101.0, 102.0]
    assert avg.get_moving_average() == pytest.approx((1.0 + 5.0 + 3.0) / 3)


def test_fast_path_and_merge_path_agree() -> None:
    clock = FakeClock(210.0)
    in_order = TelemetryRollingAverage(max_age_seconds=1000.0, clock=clock)
    out_order = TelemetryRollingAverage(max_age_seconds=1000.0, clock=clock)

    in_order.add_batch([Sample(200.0, 1.0), Sample(201.0, 2.0), Sample(202.0, 3.0)])
    # same data delivered as interleaving out-of-order batches
    out_order.add_batch([Sample(201.0, 2.0)])
    out_order.add_batch([Sample(200.0, 1.0), Sample(202.0, 3.0)])

    assert in_order.get_moving_average() == out_order.get_moving_average()
    assert [s.timestamp for s in out_order._buffer] == [200.0, 201.0, 202.0]


def test_empty_batch_still_refreshes_window() -> None:
    clock = FakeClock(100.0)
    avg = TelemetryRollingAverage(max_age_seconds=10.0, clock=clock)
    avg.add_batch([Sample(100.0, 5.0)])
    clock.advance(100.0)
    avg.add_batch([])  # empty poke should evict the aged-out sample
    assert avg.window_count() == 0


def test_resync_clears_float_drift() -> None:
    clock = FakeClock(0.0)
    avg = TelemetryRollingAverage(max_age_seconds=1.0, clock=clock)
    # churn: add and let samples age out repeatedly to accumulate add/subtract drift
    for i in range(1000):
        clock.advance(0.5)
        avg.add_batch([Sample(clock.t, 0.1)])
    avg.resync()
    expected = math.fsum(s.value for s in avg._buffer)
    assert avg._running_sum == expected


def test_explicit_now_overrides_clock() -> None:
    avg = TelemetryRollingAverage(max_age_seconds=10.0, clock=FakeClock(100.0))
    avg.add_batch([Sample(100.0, 4.0)], now=100.0)
    # ask as if it is far in the future → sample is out of window
    assert avg.get_moving_average(now=1000.0) == 0.0


def test_rejects_nonpositive_window() -> None:
    with pytest.raises(ValueError):
        TelemetryRollingAverage(max_age_seconds=0.0)
    with pytest.raises(ValueError):
        TelemetryRollingAverage(max_age_seconds=-1.0)


def test_single_loop_concurrent_adds_need_no_lock() -> None:
    """8 coroutines x 1000 adds on one asyncio loop; no lost updates, no lock.

    Each ``add_batch`` is synchronous (no ``await`` inside), so the single event loop
    serialises them even though the coroutines interleave at the ``sleep(0)``
    yield points — this is exactly why the structure needs no ``threading.Lock``
    under asyncio. (Sharing across OS threads is out of contract; guard
    externally there.)
    """
    avg = TelemetryRollingAverage(max_age_seconds=1e9, clock=lambda: 0.0)

    async def worker() -> None:
        for _ in range(1000):
            avg.add_batch([Sample(0.0, 1.0)])
            await asyncio.sleep(0)  # yield → let other coroutines interleave

    async def main() -> None:
        await asyncio.gather(*(worker() for _ in range(8)))

    asyncio.run(main())

    assert avg.window_count() == 8000
    assert avg.get_moving_average() == 1.0
