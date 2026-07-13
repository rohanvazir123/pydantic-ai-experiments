"""Tests for the corrected Redis token-bucket rate limiter.

Run from the repo root with the project venv:

    .venv/bin/python -m pytest basics/redis/rate_limiter/test_token_bucket_fixed.py -v

Uses ``fakeredis`` for an in-memory Redis that faithfully implements pipelines,
WATCH/MULTI/EXEC, and hashes — so the optimistic-locking code path is exercised
for real. Time is controlled by monkeypatching ``time.time`` inside the module
under test, making refill behavior deterministic (no ``sleep``).
"""

import fakeredis
import pytest
import token_bucket_fixed as tb


@pytest.fixture
def server() -> fakeredis.FakeServer:
    """A shared fake Redis server so multiple clients see the same keyspace."""
    return fakeredis.FakeServer()


@pytest.fixture
def client(server: fakeredis.FakeServer) -> fakeredis.FakeStrictRedis:
    """A byte-returning fake Redis client (``decode_responses=False``)."""
    return fakeredis.FakeStrictRedis(server=server, decode_responses=False)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    """Controllable clock: replaces ``time.time`` in the module under test.

    Returns a setter; call ``clock(t)`` to pin the current time to ``t`` seconds.
    """
    state = {"now": 1000.0}
    monkeypatch.setattr(tb.time, "time", lambda: state["now"])

    def _set(value: float) -> None:
        state["now"] = value

    _set(1000.0)
    return _set


def test_first_request_allowed_and_initializes_full_bucket(client) -> None:
    """A brand-new client starts with a full bucket, so the first call passes."""
    limiter = tb.RedisTokenBucketRateLimiter(client, capacity=5, refill_rate=1.0)
    assert limiter.is_allowed("alice") is True
    # After consuming 1 of 5, four tokens should remain persisted.
    state = client.hgetall("ratelimit:token_bucket:alice")
    assert float(state[b"tokens"]) == pytest.approx(4.0)


def test_burst_up_to_capacity_then_denied(client, clock) -> None:
    """Capacity requests succeed in a burst; the next is rate limited."""
    limiter = tb.RedisTokenBucketRateLimiter(client, capacity=3, refill_rate=1.0)
    # No time passes between calls, so no refill happens.
    assert [limiter.is_allowed("bob") for _ in range(3)] == [True, True, True]
    assert limiter.is_allowed("bob") is False


def test_denied_request_does_not_consume_tokens(client, clock) -> None:
    """A denied request must leave the token count unchanged."""
    limiter = tb.RedisTokenBucketRateLimiter(client, capacity=1, refill_rate=1.0)
    assert limiter.is_allowed("carol") is True   # bucket -> 0
    assert limiter.is_allowed("carol") is False  # denied
    state = client.hgetall("ratelimit:token_bucket:carol")
    assert float(state[b"tokens"]) == pytest.approx(0.0)


def test_tokens_refill_over_time(client, clock) -> None:
    """Tokens accrue at ``refill_rate`` per second while idle."""
    limiter = tb.RedisTokenBucketRateLimiter(client, capacity=5, refill_rate=2.0)
    for _ in range(5):  # drain the bucket
        limiter.is_allowed("dave")
    assert limiter.is_allowed("dave") is False   # empty
    clock(1002.0)                                 # +2s -> +4 tokens at 2/s
    assert [limiter.is_allowed("dave") for _ in range(4)] == [True] * 4
    assert limiter.is_allowed("dave") is False    # 4 tokens spent, empty again


def test_refill_is_capped_at_capacity(client, clock) -> None:
    """Idle time longer than a full refill never exceeds capacity."""
    limiter = tb.RedisTokenBucketRateLimiter(client, capacity=3, refill_rate=1.0)
    limiter.is_allowed("erin")   # consume 1 -> 2 remain, sets last_refill
    clock(2000.0)                # ~1000s idle would add 1000 tokens if uncapped
    assert [limiter.is_allowed("erin") for _ in range(3)] == [True] * 3
    assert limiter.is_allowed("erin") is False   # capped at 3, not more


def test_consume_multiple_tokens_at_once(client, clock) -> None:
    """``tokens_to_consume`` > 1 draws several tokens in one request."""
    limiter = tb.RedisTokenBucketRateLimiter(client, capacity=10, refill_rate=1.0)
    assert limiter.is_allowed("frank", tokens_to_consume=4) is True
    assert limiter.is_allowed("frank", tokens_to_consume=4) is True
    # 2 tokens left; asking for 4 must be denied without consuming.
    assert limiter.is_allowed("frank", tokens_to_consume=4) is False
    state = client.hgetall("ratelimit:token_bucket:frank")
    assert float(state[b"tokens"]) == pytest.approx(2.0)


def test_clients_are_independent(client, clock) -> None:
    """Draining one client's bucket does not affect another's."""
    limiter = tb.RedisTokenBucketRateLimiter(client, capacity=1, refill_rate=1.0)
    assert limiter.is_allowed("g1") is True
    assert limiter.is_allowed("g1") is False
    assert limiter.is_allowed("g2") is True  # separate bucket, still full


def test_ttl_is_set_for_cleanup(client, clock) -> None:
    """The bucket key gets an expiry so idle clients are reclaimed."""
    limiter = tb.RedisTokenBucketRateLimiter(client, capacity=4, refill_rate=2.0)
    limiter.is_allowed("henry")
    ttl = client.ttl("ratelimit:token_bucket:henry")
    # max_idle_ttl = int((4/2)*2) + 60 = 64
    assert ttl == 64


def test_retries_and_succeeds_on_watch_error(client, monkeypatch) -> None:
    """A WatchError from EXEC triggers a retry that ultimately succeeds.

    We wrap the client's ``pipeline`` so the first ``execute()`` raises
    ``redis.WatchError`` (simulating a concurrent writer) and later calls
    behave normally. ``is_allowed`` must swallow it, loop, and return.

    Note: real redis-py resets the pipeline inside ``execute()`` *before*
    raising ``WatchError`` (that reset is what lets ``watch()`` be re-issued on
    the next loop). The double replicates that by resetting the inner pipeline
    before raising — otherwise retry would fail with "WATCH after a MULTI".
    """
    real_pipeline = client.pipeline
    calls = {"execute": 0}

    class FlakyPipeline:
        def __init__(self, inner) -> None:
            self._inner = inner

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args) -> None:
            self._inner.__exit__(*args)

        def execute(self):
            calls["execute"] += 1
            if calls["execute"] == 1:
                self._inner.reset()  # mirror redis-py: reset before raising
                raise tb.redis.WatchError("simulated concurrent modification")
            return self._inner.execute()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(client, "pipeline", lambda *a, **k: FlakyPipeline(real_pipeline(*a, **k)))

    limiter = tb.RedisTokenBucketRateLimiter(client, capacity=5, refill_rate=1.0)
    assert limiter.is_allowed("iris") is True
    assert calls["execute"] == 2  # one failed attempt + one successful retry


def test_real_watch_abort_forces_recompute(client, server, clock, monkeypatch) -> None:
    """A genuine concurrent write between WATCH and EXEC aborts and re-reads.

    fakeredis enforces real WATCH semantics. We let a second client empty the
    bucket after our WATCH+read but before our EXEC. fakeredis then raises
    ``WatchError`` on EXEC; ``is_allowed`` must retry, re-read the now-empty
    bucket, and correctly deny the request instead of acting on stale state.
    """
    other = fakeredis.FakeStrictRedis(server=server, decode_responses=False)
    key = "ratelimit:token_bucket:jane"
    real_pipeline = client.pipeline
    calls = {"execute": 0}

    class ContendingPipeline:
        def __init__(self, inner) -> None:
            self._inner = inner

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args) -> None:
            self._inner.__exit__(*args)

        def execute(self):
            calls["execute"] += 1
            if calls["execute"] == 1:
                # Concurrent writer drains the watched key right before our EXEC.
                other.hset(key, mapping={"tokens": "0", "last_refill": "1000.0"})
            return self._inner.execute()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(client, "pipeline", lambda *a, **k: ContendingPipeline(real_pipeline(*a, **k)))

    limiter = tb.RedisTokenBucketRateLimiter(client, capacity=5, refill_rate=1.0)
    # Our stale read saw a full bucket, but the concurrent write emptied it; the
    # transaction must abort, retry, and deny based on the fresh (empty) state.
    assert limiter.is_allowed("jane") is False
    assert calls["execute"] == 2  # aborted attempt + retry


def test_clients_share_keyspace_and_respect_capacity(client, server, clock) -> None:
    """Two limiter instances on the same key never over-issue beyond capacity."""
    other = fakeredis.FakeStrictRedis(server=server, decode_responses=False)
    limiter_a = tb.RedisTokenBucketRateLimiter(client, capacity=2, refill_rate=1.0)
    limiter_b = tb.RedisTokenBucketRateLimiter(other, capacity=2, refill_rate=1.0)

    # Frozen clock (via the ``clock`` fixture) means no refill between calls.
    allowed = sum(
        [
            limiter_a.is_allowed("shared"),
            limiter_b.is_allowed("shared"),
            limiter_a.is_allowed("shared"),
            limiter_b.is_allowed("shared"),
        ]
    )
    assert allowed == 2  # capacity respected across both clients
