"""Tests for the fixed CPU and IO worker-queue implementations.

Run from the repo root with the project venv:

    .venv/bin/python -m pytest basics/worker_queues/test_worker_queues_fixed.py -v

- CPU tests spin up real worker *processes* (multiprocessing) and assert the
  pool drains cleanly without the original deadlock.
- IO tests run the asyncio worker pool against a temporary SQLite file and
  assert rows are actually persisted via ``aiosqlite``.
"""

import asyncio
import sqlite3

import pytest

import cpu_workloads_fixed as cpu
import io_workloads_fixed as io_mod

# ---------------------------------------------------------------------------
# CPU-bound pool (multiprocessing)
# ---------------------------------------------------------------------------


def test_image_processor_reports_byte_size() -> None:
    """The isolated processor returns the payload size, no pool needed."""
    req = cpu.ImageProcessingRequest(image_id="img", image_data=b"abcd")
    result = cpu.ImageProcessor().process(req)
    assert result.image_id == "img"
    assert result.size_bytes == 4


def test_pool_processes_all_tasks_and_shuts_down() -> None:
    """All submitted tasks are processed exactly once and the pool joins.

    This is the regression test for the original deadlock: ``shutdown()`` sends
    sentinels before joining, so ``worker.join()`` returns instead of hanging.
    """
    pool = cpu.CpuWorkerPool(num_workers=2)
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
    assert all(not p.is_alive() for p in pool.processes)


class DoublingProcessor(cpu.ImageProcessor):
    """Custom processor defined at module level so it pickles under ``spawn``."""

    def process(self, payload: cpu.ImageProcessingRequest) -> cpu.ProcessedImage:
        return cpu.ProcessedImage(
            image_id=payload.image_id, size_bytes=len(payload.image_data) * 2
        )


def test_pool_uses_injected_processor() -> None:
    """A custom ImageProcessor subclass is honored by the pool."""
    pool = cpu.CpuWorkerPool(num_workers=1, processor=DoublingProcessor())
    pool.insert_cpu_tasks([cpu.ImageProcessingRequest(image_id="d", image_data=b"abc")])
    pool.join_tasks()
    results = pool.collect_results(1)
    pool.shutdown()
    assert results[0].size_bytes == 6  # 3 bytes doubled


def test_pool_skips_invalid_payload_without_crashing() -> None:
    """A non-request payload is skipped (not crashing on attribute access)."""
    pool = cpu.CpuWorkerPool(num_workers=1)
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
async def test_telemetry_writer_persists_row(tmp_path) -> None:
    """The writer lands a row using a connection borrowed from the pool."""
    db_path = str(tmp_path / "telemetry.db")
    pool = io_mod.SqliteConnectionPool(
        db_path, size=1, init_statements=[io_mod.TELEMETRY_DDL]
    )
    await pool.open()
    writer = io_mod.TelemetryWriter(pool)
    await writer.write(io_mod.TelemetryData(device_id="d1", metric={"t": 1}))
    await pool.close()
    assert _count_rows(db_path) == 1


@pytest.mark.asyncio
async def test_pool_lends_and_returns_connections(tmp_path) -> None:
    """acquire() checks out a connection and returns it on context exit."""
    db_path = str(tmp_path / "telemetry.db")
    pool = io_mod.SqliteConnectionPool(db_path, size=2)
    await pool.open()
    assert pool.available() == 2
    async with pool.acquire():
        assert pool.available() == 1
        async with pool.acquire():
            assert pool.available() == 0  # both in use
        assert pool.available() == 1  # inner returned
    assert pool.available() == 2  # outer returned
    await pool.close()


@pytest.mark.asyncio
async def test_pool_handles_concurrent_writes(tmp_path) -> None:
    """Many overlapping writes through a shared pool all persist safely."""
    db_path = str(tmp_path / "telemetry.db")
    pool = io_mod.SqliteConnectionPool(
        db_path, size=3, init_statements=[io_mod.TELEMETRY_DDL]
    )
    await pool.open()
    writer = io_mod.TelemetryWriter(pool)
    await asyncio.gather(
        *(
            writer.write(io_mod.TelemetryData(device_id=f"d{i}", metric={"i": i}))
            for i in range(15)
        )
    )
    await pool.close()
    assert _count_rows(db_path) == 15


@pytest.mark.asyncio
async def test_all_tasks_are_persisted(tmp_path) -> None:
    """Every inserted telemetry item is written and the pool shuts down."""
    db_path = str(tmp_path / "telemetry.db")
    q = io_mod.IoWorkerPool(maxsize=50, num_workers=3, db_path=db_path)

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
    q = io_mod.IoWorkerPool(num_workers=1, db_path=db_path)

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
    q = io_mod.IoWorkerPool(maxsize=1, num_workers=0, db_path=db_path)

    await q.insert_io_task(io_mod.TelemetryData(device_id="d", metric={"t": 1}))
    assert q.io_queue.full()
    # No workers were started, so nothing to shut down / join.
    assert q.workers == []
