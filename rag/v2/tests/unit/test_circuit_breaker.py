"""Unit tests for knowledge.bus.circuit_breaker.

Uses fakeredis — no live Redis required.
"""

import asyncio
import pytest
import pytest_asyncio
import fakeredis.aioredis as fakeredis

from knowledge.bus.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    _STATE_CLOSED,
    _STATE_OPEN,
    _STATE_HALF_OPEN,
)


def make_cb(redis, **kwargs) -> CircuitBreaker:
    defaults = dict(
        open_threshold=3,
        window_seconds=60,
        probe_interval_s=30,
        consecutive_success_threshold=2,
    )
    defaults.update(kwargs)
    return CircuitBreaker("test_service", redis, **defaults)


@pytest_asyncio.fixture
async def redis():
    return fakeredis.FakeRedis(decode_responses=False)


@pytest_asyncio.fixture
async def cb(redis):
    return make_cb(redis)


# ── Happy path ────────────────────────────────────────────────────────────────

class TestClosedState:
    @pytest.mark.asyncio
    async def test_call_passes_through_when_closed(self, cb):
        async def succeed():
            return 42
        result = await cb.call(succeed())
        assert result == 42

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self, cb):
        state = await cb._get_state()
        assert state == _STATE_CLOSED

    @pytest.mark.asyncio
    async def test_success_keeps_circuit_closed(self, cb):
        async def ok(): return "ok"
        await cb.call(ok())
        assert await cb._get_state() == _STATE_CLOSED


# ── Opening the circuit ───────────────────────────────────────────────────────

class TestOpenTransition:
    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self, cb):
        async def fail():
            raise ConnectionError("boom")

        for _ in range(3):  # open_threshold = 3
            with pytest.raises(ConnectionError):
                await cb.call(fail())

        assert await cb._get_state() == _STATE_OPEN

    @pytest.mark.asyncio
    async def test_below_threshold_stays_closed(self, cb):
        async def fail():
            raise ConnectionError("boom")

        for _ in range(2):  # one below threshold
            with pytest.raises(ConnectionError):
                await cb.call(fail())

        assert await cb._get_state() == _STATE_CLOSED

    @pytest.mark.asyncio
    async def test_open_raises_circuit_open_error(self, cb):
        async def fail():
            raise ConnectionError("boom")

        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(fail())

        async def ok(): return "ok"
        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.call(ok())
        assert exc_info.value.service == "test_service"

    @pytest.mark.asyncio
    async def test_circuit_open_error_has_retry_after(self, cb):
        async def fail():
            raise RuntimeError("boom")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(fail())

        async def ok(): pass
        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.call(ok())
        assert exc_info.value.retry_after_s > 0


# ── Half-open and recovery ────────────────────────────────────────────────────

class TestHalfOpenTransition:
    @pytest.mark.asyncio
    async def test_half_open_after_probe_interval(self, redis):
        cb = make_cb(redis, open_threshold=1, probe_interval_s=0)

        async def fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await cb.call(fail())

        assert await cb._get_state() == _STATE_OPEN

        # probe_interval_s=0 means probe is immediately allowed
        async def ok(): return "good"
        # First call transitions to HALF-OPEN and runs probe
        await cb.call(ok())
        # State should be HALF-OPEN (need 2 consecutive successes to close)
        state = await cb._get_state()
        assert state in (_STATE_HALF_OPEN, _STATE_CLOSED)

    @pytest.mark.asyncio
    async def test_closes_after_consecutive_successes(self, redis):
        cb = make_cb(
            redis,
            open_threshold=1,
            probe_interval_s=0,
            consecutive_success_threshold=2,
        )

        async def fail():
            raise RuntimeError()

        with pytest.raises(RuntimeError):
            await cb.call(fail())

        async def ok(): return "ok"
        await cb.call(ok())   # probe → HALF-OPEN, success 1
        await cb.call(ok())   # success 2 → CLOSED
        assert await cb._get_state() == _STATE_CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self, redis):
        cb = make_cb(redis, open_threshold=1, probe_interval_s=0)

        async def fail():
            raise RuntimeError()

        with pytest.raises(RuntimeError):
            await cb.call(fail())

        # Probe attempt — also fails → should re-open
        with pytest.raises(RuntimeError):
            await cb.call(fail())

        assert await cb._get_state() == _STATE_OPEN


# ── State is shared in Redis ──────────────────────────────────────────────────

class TestSharedState:
    @pytest.mark.asyncio
    async def test_two_instances_share_state(self, redis):
        """Opening on cb1 must block cb2 (same Redis)."""
        cb1 = make_cb(redis, open_threshold=1, probe_interval_s=60)
        cb2 = make_cb(redis, open_threshold=1, probe_interval_s=60)

        async def fail():
            raise RuntimeError()

        with pytest.raises(RuntimeError):
            await cb1.call(fail())

        async def ok(): return "ok"
        with pytest.raises(CircuitOpenError):
            await cb2.call(ok())
