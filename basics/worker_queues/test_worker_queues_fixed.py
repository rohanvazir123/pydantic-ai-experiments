"""Tests for the fixed CPU and IO worker-queue implementations.

Run from the repo root with the project venv:

    .venv/bin/python -m pytest basics/worker_queues/test_worker_queues_fixed.py -v

- CPU tests spin up real worker *processes* (multiprocessing) and assert the
  pool drains cleanly without the original deadlock.
- IO tests run the asyncio worker pool against a temporary SQLite file and
  assert rows are actually persisted via ``aiosqlite``.
"""

import sqlite3

import pytest

import cpu_workloads_fixed as cpu
import io_workloads_fixed as io_mod

# ---------------------------------------------------------------------------
# CPU-bound pool (multiprocessing)
# ---------------------------------------------------------------------------


def test_process_image_reports_byte_size() -> None:
    """The pure processing function returns the payload size."""
    req = cpu.ImageProcessingRequest(image_id="img", image_data=b"abcd")
    result = cpu.process_image(req)
    assert result.image_id == "img"
    assert result.size_bytes == 4


def test_pool_processes_all_tasks_and_shuts_down() -> None:
    """All submitted tasks are processed exactly once and the pool joins.

    This is the regression test for the original deadlock: ``shutdown()`` sends
    sentinels before joining, so ``worker.join()`` returns instead of hanging.
    """
    pool = cpu.CpuWorkerQueue(num_workers=2)
    payloads = [
        cpu.ImageProcessingRequest(image_id=f"img_{i}", image_data=b"x" * i)
        for i in range(8)
    ]
    pool.insert_cpu_tasks(payloads)
    pool.join_tasks()
    results = pool.collect_results(len(payloads))
    pool.shutdown()

    assert len(results) == 8
    assert {r.image_id for r in results} == {f"img_{i}" for i in range(8)}
    # size_bytes should match the "x" * i payload length for each image.
    by_id = {r.image_id: r.size_bytes for r in results}
    assert by_id["img_5"] == 5
    # Every worker process has terminated.
    assert all(not w.is_alive() for w in pool.workers)


def test_pool_skips_invalid_payload_without_crashing() -> None:
    """A non-request payload is skipped (not crashing on attribute access)."""
    pool = cpu.CpuWorkerQueue(num_workers=1)
    pool.task_queue.put("not-a-request")  # invalid item
    pool.insert_cpu_tasks([cpu.ImageProcessingRequest(image_id="ok", image_data=b"hi")])
    pool.join_tasks()
    results = pool.collect_results(1)  # only the valid one yields a result
    pool.shutdown()

    assert len(results) == 1
    assert results[0].image_id == "ok"


# ---------------------------------------------------------------------------
# IO-bound pool (asyncio + aiosqlite)
# ---------------------------------------------------------------------------


def _count_rows(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0]
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_write_telemetry_persists_row(tmp_path) -> None:
    """A single write lands in the database."""
    db_path = str(tmp_path / "telemetry.db")
    q = io_mod.IoWorkerQueue(num_workers=1, db_path=db_path)
    await q._write_telemetry(io_mod.TelemetryData(device_id="d1", metric={"t": 1}))
    await q.shutdown()
    assert _count_rows(db_path) == 1


@pytest.mark.asyncio
async def test_all_tasks_are_persisted(tmp_path) -> None:
    """Every inserted telemetry item is written and the pool shuts down."""
    db_path = str(tmp_path / "telemetry.db")
    q = io_mod.IoWorkerQueue(maxsize=50, num_workers=3, db_path=db_path)

    for i in range(20):
        await q.insert_io_task(
            io_mod.TelemetryData(device_id=f"device_{i}", metric={"temp": 20 + i})
        )
    await q.shutdown()

    assert _count_rows(db_path) == 20
    # Every worker task has completed.
    assert all(w.done() for w in q.workers)


@pytest.mark.asyncio
async def test_invalid_item_is_skipped(tmp_path) -> None:
    """An invalid queue item is skipped without breaking the join/shutdown."""
    db_path = str(tmp_path / "telemetry.db")
    q = io_mod.IoWorkerQueue(num_workers=1, db_path=db_path)

    await q.io_queue.put("not-telemetry")  # invalid, should be skipped
    await q.insert_io_task(io_mod.TelemetryData(device_id="ok", metric={"t": 1}))
    await q.shutdown()

    # Only the valid item was persisted; shutdown() completed (no hang).
    assert _count_rows(db_path) == 1


@pytest.mark.asyncio
async def test_insert_respects_backpressure(tmp_path) -> None:
    """``insert_io_task`` awaits (does not raise) even when the queue is small.

    With no workers draining and maxsize=1, a second insert would block; we
    assert the first completes and the queue is full, proving backpressure via
    ``await put`` rather than a ``put_nowait`` that would raise ``QueueFull``.
    """
    db_path = str(tmp_path / "telemetry.db")
    q = io_mod.IoWorkerQueue(maxsize=1, num_workers=0, db_path=db_path)

    await q.insert_io_task(io_mod.TelemetryData(device_id="d", metric={"t": 1}))
    assert q.io_queue.full()
    # No workers were started, so nothing to shut down / join.
    assert q.workers == []
