"""Redis-backed circuit breaker.

State is stored in Redis so all API pods and workers share the same view.
A circuit that opens on one pod is immediately open on all pods.

State machine:
    CLOSED   → normal operation; failure counter maintained
    OPEN     → all calls blocked immediately; probe timer running
    HALF-OPEN → one probe call allowed; success→CLOSED, failure→OPEN

Redis keys (all scoped to `name`):
    cb:{name}:state      — "CLOSED" | "OPEN" | "HALF-OPEN"
    cb:{name}:failures   — integer counter (expires after WINDOW_SECONDS)
    cb:{name}:opened_at  — Unix timestamp float (when circuit opened)
    cb:{name}:half_successes — integer counter (consecutive successes in HALF-OPEN)
"""

import logging
import time
from typing import Awaitable, TypeVar

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

T = TypeVar("T")

_STATE_CLOSED    = "CLOSED"
_STATE_OPEN      = "OPEN"
_STATE_HALF_OPEN = "HALF-OPEN"

# Defaults — override per-instance via constructor
_OPEN_THRESHOLD               = 5    # failures in WINDOW_SECONDS before opening
_WINDOW_SECONDS               = 60   # failure counting window
_PROBE_INTERVAL_S             = 30   # seconds to wait before HALF-OPEN probe
_CONSECUTIVE_SUCCESS_THRESHOLD = 2   # successes in HALF-OPEN to close circuit


class CircuitOpenError(Exception):
    """Raised when a call is blocked by an open circuit breaker."""

    def __init__(self, service: str, retry_after_s: float) -> None:
        self.service = service
        self.retry_after_s = retry_after_s
        super().__init__(
            f"Circuit '{service}' is OPEN. Retry after {retry_after_s:.0f}s."
        )


class CircuitBreaker:
    """Redis-backed circuit breaker for a named external service.

    Usage:
        cb = CircuitBreaker("age_graph", redis_client)
        result = await cb.call(some_coroutine())
    """

    def __init__(
        self,
        name: str,
        redis: aioredis.Redis,
        open_threshold: int = _OPEN_THRESHOLD,
        window_seconds: int = _WINDOW_SECONDS,
        probe_interval_s: float = _PROBE_INTERVAL_S,
        consecutive_success_threshold: int = _CONSECUTIVE_SUCCESS_THRESHOLD,
    ) -> None:
        self._name = name
        self._redis = redis
        self._open_threshold = open_threshold
        self._window_seconds = window_seconds
        self._probe_interval_s = probe_interval_s
        self._consecutive_success_threshold = consecutive_success_threshold

    # ── Redis key helpers ─────────────────────────────────────────────────────

    @property
    def _state_key(self) -> str:
        return f"cb:{self._name}:state"

    @property
    def _failures_key(self) -> str:
        return f"cb:{self._name}:failures"

    @property
    def _opened_at_key(self) -> str:
        return f"cb:{self._name}:opened_at"

    @property
    def _half_successes_key(self) -> str:
        return f"cb:{self._name}:half_successes"

    # ── State accessors ───────────────────────────────────────────────────────

    async def _get_state(self) -> str:
        raw = await self._redis.get(self._state_key)
        return raw.decode() if raw else _STATE_CLOSED

    async def _set_state(self, state: str) -> None:
        await self._redis.set(self._state_key, state)

    async def _probe_remaining_s(self) -> float:
        """Seconds until the OPEN→HALF-OPEN probe is allowed."""
        raw = await self._redis.get(self._opened_at_key)
        if raw is None:
            return 0.0
        opened_at = float(raw)
        elapsed = time.time() - opened_at
        remaining = self._probe_interval_s - elapsed
        return max(0.0, remaining)

    # ── Transition helpers ────────────────────────────────────────────────────

    async def _open(self) -> None:
        pipe = self._redis.pipeline()
        pipe.set(self._state_key, _STATE_OPEN)
        pipe.set(self._opened_at_key, str(time.time()))
        pipe.delete(self._half_successes_key)
        await pipe.execute()
        logger.warning("CircuitBreaker '%s' → OPEN", self._name)

    async def _close(self) -> None:
        pipe = self._redis.pipeline()
        pipe.set(self._state_key, _STATE_CLOSED)
        pipe.delete(self._failures_key)
        pipe.delete(self._opened_at_key)
        pipe.delete(self._half_successes_key)
        await pipe.execute()
        logger.info("CircuitBreaker '%s' → CLOSED", self._name)

    async def _record_failure(self) -> None:
        count = await self._redis.incr(self._failures_key)
        # Set expiry only on first increment to preserve the window
        if count == 1:
            await self._redis.expire(self._failures_key, self._window_seconds)
        if count >= self._open_threshold:
            await self._open()

    async def _record_success(self) -> None:
        state = await self._get_state()
        if state == _STATE_HALF_OPEN:
            successes = await self._redis.incr(self._half_successes_key)
            if successes >= self._consecutive_success_threshold:
                await self._close()

    # ── Public interface ──────────────────────────────────────────────────────

    async def call(self, coro: Awaitable[T]) -> T:
        """Execute `coro`, enforcing circuit breaker state.

        Raises CircuitOpenError immediately when the circuit is OPEN.
        In HALF-OPEN, allows one probe through; on success closes the circuit;
        on failure re-opens it.
        """
        state = await self._get_state()

        if state == _STATE_OPEN:
            remaining = await self._probe_remaining_s()
            if remaining > 0:
                # Close the unawaited coroutine to silence "coroutine never awaited" warning
                import inspect
                if inspect.iscoroutine(coro):
                    coro.close()
                raise CircuitOpenError(self._name, remaining)
            # Probe interval elapsed → transition to HALF-OPEN
            await self._set_state(_STATE_HALF_OPEN)
            logger.info("CircuitBreaker '%s' → HALF-OPEN (probe)", self._name)
            state = _STATE_HALF_OPEN

        try:
            result = await coro
            await self._record_success()
            return result
        except Exception as exc:
            state_now = await self._get_state()
            if state_now == _STATE_HALF_OPEN:
                await self._open()
            else:
                await self._record_failure()
            raise exc
