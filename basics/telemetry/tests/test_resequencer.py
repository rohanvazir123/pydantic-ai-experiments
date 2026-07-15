"""Tests for the out-of-order telemetry resequencer.

Run from the repo root with the project venv:

    .venv/bin/python -m pytest basics/telemetry/tests/test_resequencer.py -v
"""

import time

import pytest
from base import JobStatus
from resequencer import Resequencer, ResequencingWorkerPool, TelemetryFrame


def frame(seq: int, device_id: str = "drone-1") -> TelemetryFrame:
    return TelemetryFrame(device_id=device_id, seq=seq, payload={"alt_m": float(seq)})


def seqs(resequencer: Resequencer, device_id: str = "drone-1") -> list[int]:
    return [f.seq for f in resequencer.emitted[device_id]]


def test_in_order_submits_emit_immediately() -> None:
    resequencer = Resequencer()
    for i in range(5):
        resequencer.submit(frame(i))
    assert seqs(resequencer) == [0, 1, 2, 3, 4]


def test_gap_fills_on_a_later_submit() -> None:
    """A blocked device emits NOTHING and loses nothing; the frame that fills
    the gap releases everything queued behind it, on its own submit.

    `_emit_ready` never waits for that frame -- it returns immediately, leaving
    the heap intact, and the next submit is what makes progress.
    """
    resequencer = Resequencer()
    resequencer.submit(frame(2))
    resequencer.submit(frame(3))
    assert resequencer.emitted == {}  # blocked on seq=0: withheld, not dropped

    resequencer.submit(frame(0))
    assert seqs(resequencer) == [0]  # 0 released; 2 and 3 still blocked on seq=1

    resequencer.submit(frame(1))  # this submit releases 1, 2 and 3 together
    assert seqs(resequencer) == [0, 1, 2, 3]
    assert resequencer.gaps == []  # nothing was ever declared lost


def test_stale_frame_is_dropped_at_submit() -> None:
    """A seq below `expected` is rejected before it reaches the heap."""
    resequencer = Resequencer()
    resequencer.submit(frame(0))
    resequencer.submit(frame(1))
    resequencer.submit(frame(0))  # already emitted
    resequencer.submit(frame(2))
    resequencer.submit(frame(2))  # already emitted
    assert seqs(resequencer) == [0, 1, 2]
    assert len(resequencer.dropped) == 2


def test_duplicate_of_a_still_buffered_frame_is_emitted_once() -> None:
    """The case `seq < expected` can't catch: a dup that arrives while its twin
    is still buffered. Both sit in the heap; `_emit_ready` drops the loser.
    """
    resequencer = Resequencer()
    resequencer.submit(frame(0))  # emitted, expected -> 1
    resequencer.submit(frame(2))  # early, buffered
    resequencer.submit(frame(2))  # duplicate, also buffered
    resequencer.submit(frame(1))  # emits 1, then 2 once, drops the dup
    assert seqs(resequencer) == [0, 1, 2]
    assert len(resequencer.dropped) == 1


def test_gap_times_out_even_though_buffer_never_fills() -> None:
    """The size bound alone can't fire here: only 2 frames are ever withheld,
    nowhere near max_buffer. Without the time bound they'd stall forever.
    """
    resequencer = Resequencer(max_buffer=100, max_delay=0.05)
    resequencer.submit(frame(0))
    resequencer.submit(frame(2))  # seq=1 never arrives; 2 is withheld
    assert seqs(resequencer) == [0]  # still waiting, buffer nowhere near full

    time.sleep(0.06)  # outlive max_delay
    resequencer.submit(frame(3))  # any submit re-checks the clock

    assert seqs(resequencer) == [0, 2, 3]
    assert resequencer.gaps == [("drone-1", 1, 2)]


def test_close_releases_the_tail_left_after_the_last_gap() -> None:
    """End of stream: no further submit will ever test the bounds, so whatever
    trails the last gap is only released by close().
    """
    resequencer = Resequencer(max_buffer=100, max_delay=100.0)
    resequencer.submit(frame(0))
    resequencer.submit(frame(2))  # seq=1 never arrives
    assert seqs(resequencer) == [0]  # withheld: both bounds set out of reach

    resequencer.close()

    assert seqs(resequencer) == [0, 2]
    assert resequencer.gaps == [("drone-1", 1, 2)]


def test_permanently_missing_seq_forces_advance_once_buffer_fills() -> None:
    resequencer = Resequencer(max_buffer=3, max_delay=100.0)
    resequencer.submit(frame(0))
    # seq=1 never arrives; 2..5 pile up in the buffer instead
    for i in (2, 3, 4, 5):
        resequencer.submit(frame(i))
    # buffer exceeded max_buffer -> gap skipped, everything buffered released
    assert resequencer.gaps == [("drone-1", 1, 2)]
    assert seqs(resequencer) == [0, 2, 3, 4, 5]


def test_devices_are_resequenced_independently() -> None:
    resequencer = Resequencer()
    resequencer.submit(frame(1, device_id="drone-A"))
    resequencer.submit(frame(0, device_id="drone-B"))
    resequencer.submit(frame(0, device_id="drone-A"))
    resequencer.submit(frame(1, device_id="drone-B"))
    assert seqs(resequencer, "drone-A") == [0, 1]
    assert seqs(resequencer, "drone-B") == [0, 1]


@pytest.mark.asyncio
async def test_worker_pool_resequences_despite_concurrent_out_of_order_completion() -> None:
    """Frames go in ordered; 8 concurrent workers scramble completion; they
    still come out ordered. Both bounds are held clear of what normal jitter
    needs, so a clean run emits everything with no gaps.
    """
    resequencer = Resequencer(max_buffer=25, max_delay=100.0)
    pool = ResequencingWorkerPool(num_workers=8, resequencer=resequencer)
    for i in range(20):
        await pool.insert_job(frame(i))
    await pool.join_tasks()
    await pool.shutdown()

    assert seqs(resequencer) == list(range(20))
    assert resequencer.gaps == []
    outcomes = pool.collect_results()
    assert all(s == JobStatus.DONE for s in outcomes.values())
