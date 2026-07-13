"""Tests for the fixed CPU and IO worker-queue implementations.

Run from the repo root with the project venv:

    .venv/bin/python -m pytest basics/telemetry/tests/test_worker_queues_fixed.py -v

- CPU tests spin up real worker *processes* (multiprocessing) and assert the
  pool drains cleanly without the original deadlock.
- IO tests run the asyncio worker pool against an in-memory SQLite DB (via
  SQLAlchemy) and assert rows are actually persisted.
"""

import asyncio

import cpu_workloads_fixed as cpu
import io_workloads_fixed as io_mod
import pytest

# ---------------------------------------------------------------------------
# CPU-bound pool (multiprocessing)
# ---------------------------------------------------------------------------


def test_job_processes_itself() -> None:
    """A job returns its own result via process(), no pool needed."""
    result = cpu.ImageProcessingRequest(image_id="img", image_data=b"abcd").process()
    assert result.image_id == "img"
    assert result.size_bytes == 4


def test_pool_processes_all_tasks_and_shuts_down() -> None:
    """All submitted tasks are processed exactly once and the pool joins.

    Regression test for the original deadlock: ``shutdown()`` sends sentinels
    before joining, so ``worker.join()`` returns instead of hanging.
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
    by_id = {r.image_id: r.size_bytes for r in results}
    assert by_id["img_5"] == 5
    assert all(not p.is_alive() for p in pool.processes)


class DoublingRequest(cpu.ImageProcessingRequest):
    """A job subclass with different behaviour; module-level so it pickles under spawn."""

    def process(
        self, sinks: "cpu.Mapping[str, cpu.ResultSink] | None" = None
    ) -> cpu.ProcessedImage:
        result = cpu.ProcessedImage(
            image_id=self.image_id, size_bytes=len(self.image_data) * 2
        )
        if sinks is not None:
            sinks[self.result_sink].emit(result)
        return result


def test_pool_dispatches_to_job_process() -> None:
    """The pool runs each job's own process(); a subclass overrides the behaviour."""
    pool = cpu.CpuWorkerPool(num_workers=1)
    pool.insert_cpu_tasks([DoublingRequest(image_id="d", image_data=b"abc")])
    pool.join_tasks()
    results = pool.collect_results(1)
    pool.shutdown()
    assert results[0].size_bytes == 6  # 3 bytes doubled


def test_file_sink_persists_results() -> None:
    """Jobs routed to the ``file`` sink are readable back from disk."""
    pool = cpu.CpuWorkerPool(num_workers=2)
    payloads = [
        cpu.ImageProcessingRequest(image_id=f"img_{i}", image_data=b"x" * i, result_sink="file")
        for i in range(6)
    ]
    pool.insert_cpu_tasks(payloads)
    pool.join_tasks()
    pool.collect_results(len(payloads))
    stored = cpu.read_results(pool.result_path)
    pool.shutdown()

    assert {r.image_id for r in stored} == {f"img_{i}" for i in range(6)}


def test_http_sink_receives_results() -> None:
    """Jobs routed to the ``http`` sink land in the outbox, not the file."""
    pool = cpu.CpuWorkerPool(num_workers=2)
    payloads = [
        cpu.ImageProcessingRequest(image_id=f"img_{i}", image_data=b"x", result_sink="http")
        for i in range(4)
    ]
    pool.insert_cpu_tasks(payloads)
    pool.join_tasks()
    pool.collect_results(len(payloads))
    sent = list(pool.http_outbox)
    persisted = cpu.read_results(pool.result_path)
    pool.shutdown()

    assert len(sent) == 4
    assert persisted == []  # nothing went to the file sink


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


def test_cpu_job_status_reaches_done() -> None:
    """Each submitted job is tracked QUEUED -> ... -> DONE across processes."""
    pool = cpu.CpuWorkerPool(num_workers=2)
    jobs = [
        cpu.ImageProcessingRequest(image_id=f"j{i}", image_data=b"x") for i in range(4)
    ]
    for job in jobs:
        pool.insert_job(job)
    pool.join_tasks()
    pool.collect_results(len(jobs))
    statuses = [pool.get_job_status(job.job_id) for job in jobs]
    pool.shutdown()

    assert all(s == cpu.JobStatus.DONE for s in statuses)


def test_cpu_cancel_unknown_or_completed_returns_false() -> None:
    """A finished or unknown job can't be cancelled."""
    pool = cpu.CpuWorkerPool(num_workers=1)
    job = cpu.ImageProcessingRequest(image_id="ok", image_data=b"hi")
    pool.insert_job(job)
    pool.join_tasks()
    pool.collect_results(1)
    cancel_completed = pool.cancel_job(job.job_id)  # already DONE
    cancel_unknown = pool.cancel_job("does-not-exist")
    status = pool.get_job_status(job.job_id)
    pool.shutdown()

    assert cancel_completed is False
    assert cancel_unknown is False
    assert status == cpu.JobStatus.DONE


# ---------------------------------------------------------------------------
# IO-bound pool (asyncio + SQLAlchemy / in-memory SQLite)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_persists_row() -> None:
    """The writer lands a row via SQLAlchemy."""
    engine = io_mod.make_engine()
    writer = io_mod.TelemetryWriter(engine)
    await writer.write(io_mod.TelemetryData(device_id="d1", metric={"t": 1}))
    assert io_mod.count_rows(engine) == 1


@pytest.mark.asyncio
async def test_concurrent_writes_all_persist() -> None:
    """Overlapping writes through one writer all persist."""
    engine = io_mod.make_engine()
    writer = io_mod.TelemetryWriter(engine)
    await asyncio.gather(
        *(
            writer.write(io_mod.TelemetryData(device_id=f"d{i}", metric={"i": i}))
            for i in range(15)
        )
    )
    assert io_mod.count_rows(engine) == 15


@pytest.mark.asyncio
async def test_all_tasks_are_persisted() -> None:
    """Every inserted telemetry item is written and the pool shuts down."""
    q = io_mod.IoWorkerPool(maxsize=50, num_workers=3)
    for i in range(20):
        await q.insert_job(
            io_mod.TelemetryData(device_id=f"device_{i}", metric={"temp": 20 + i})
        )
    await q.shutdown()

    assert io_mod.count_rows(q.engine) == 20
    assert all(w.done() for w in q.workers)


@pytest.mark.asyncio
async def test_invalid_item_is_skipped() -> None:
    """An invalid queue item is skipped without breaking the join/shutdown."""
    q = io_mod.IoWorkerPool(num_workers=1)
    await q.io_queue.put("not-telemetry")  # invalid, should be skipped
    await q.insert_job(io_mod.TelemetryData(device_id="ok", metric={"t": 1}))
    await q.shutdown()

    assert io_mod.count_rows(q.engine) == 1


@pytest.mark.asyncio
async def test_insert_respects_backpressure() -> None:
    """``insert_job`` awaits (does not raise) even when the queue is full.

    With no workers draining and maxsize=1, a second insert would block; we
    assert the first completes and the queue is full — backpressure via
    ``await put`` rather than a ``put_nowait`` that would raise ``QueueFull``.
    """
    q = io_mod.IoWorkerPool(maxsize=1, num_workers=0)
    await q.insert_job(io_mod.TelemetryData(device_id="d", metric={"t": 1}))
    assert q.io_queue.full()
    assert q.workers == []


@pytest.mark.asyncio
async def test_io_job_status_reaches_done() -> None:
    """A submitted job ends up DONE once the workers drain and shut down."""
    q = io_mod.IoWorkerPool(num_workers=2)
    job = io_mod.TelemetryData(device_id="d", metric={"t": 1})
    await q.insert_job(job)
    await q.shutdown()

    assert q.get_job_status(job.job_id) == io_mod.JobStatus.DONE


@pytest.mark.asyncio
async def test_io_cancel_pending_job() -> None:
    """With no workers draining, a QUEUED job can be cancelled deterministically."""
    q = io_mod.IoWorkerPool(maxsize=10, num_workers=0)
    job = io_mod.TelemetryData(device_id="d", metric={"t": 1})

    await q.insert_job(job)
    assert q.get_job_status(job.job_id) == io_mod.JobStatus.QUEUED

    assert q.cancel_job(job.job_id) is True
    assert q.get_job_status(job.job_id) == io_mod.JobStatus.CANCELLED
    assert q.cancel_job(job.job_id) is False        # already cancelled
    assert q.cancel_job("unknown") is False         # never seen
    assert q.get_job_status("unknown") == io_mod.JobStatus.UNKNOWN
    assert q.workers == []
