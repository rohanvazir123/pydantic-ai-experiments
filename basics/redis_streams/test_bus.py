"""Tests for RedisStreamBus — run on fakeredis, no external services.

    uv run pytest
"""

from __future__ import annotations

import fakeredis
import pytest

from bus import RedisStreamBus


@pytest.fixture
async def client():
    c = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield c
    await c.aclose()


@pytest.fixture
async def bus(client):
    b = RedisStreamBus(client=client, stream="s", group="g", max_attempts=3)
    await b.ensure_group()
    return b


async def test_happy_path_acks_everything(bus, client):
    seen: list[str] = []

    async def handler(_id, fields):
        seen.append(fields["k"])

    await bus.publish({"k": "a"})
    await bus.publish({"k": "b"})
    handled = await bus.consume_once("c1", handler, block_ms=50)

    assert handled == 2
    assert seen == ["a", "b"]
    pending = await client.xpending("s", "g")
    assert pending["pending"] == 0  # all acked


async def test_failed_message_stays_pending_then_reclaimed(bus, client):
    attempts: dict[str, int] = {}

    async def handler(_id, fields):
        k = fields["k"]
        attempts[k] = attempts.get(k, 0) + 1
        if attempts[k] == 1:  # fail the first delivery only
            raise RuntimeError("boom")

    await bus.publish({"k": "x"})
    await bus.consume_once("c1", handler, block_ms=50)

    # left unacked in c1's PEL
    assert (await client.xpending("s", "g"))["pending"] == 1

    # another consumer reclaims it and succeeds on the second attempt
    reclaimed = await bus.reclaim_once("c2", handler, min_idle_ms=0)
    assert reclaimed == 1
    assert attempts["x"] == 2
    assert (await client.xpending("s", "g"))["pending"] == 0


async def test_poison_message_goes_to_dlq(bus, client):
    async def always_fails(_id, _fields):
        raise RuntimeError("poison")

    await bus.publish({"k": "bad"})
    await bus.consume_once("c1", always_fails, block_ms=50)  # attempt 1, stays pending

    # reclaim repeatedly; max_attempts=3 -> on the attempt that hits the cap it is DLQ'd
    for _ in range(5):
        if (await client.xpending("s", "g"))["pending"] == 0:
            break
        await bus.reclaim_once("c2", always_fails, min_idle_ms=0)

    assert (await client.xpending("s", "g"))["pending"] == 0  # drained from the group
    dlq = await client.xrange("s:dlq")
    assert len(dlq) == 1
    _id, fields = dlq[0]
    assert fields["k"] == "bad"
    assert "_failed_id" in fields


async def test_idempotency_guard_runs_effect_once(bus, client):
    effects: list[str] = []

    async def handler(_id, fields):
        k = fields["k"]
        # first delivery: record effect but then "crash" before ack
        if await client.set(f"done:{k}", "1", nx=True, ex=60):
            effects.append(k)
        if len(effects) == 1 and k not in getattr(handler, "_acked", set()):
            handler._acked = {k}  # type: ignore[attr-defined]
            raise RuntimeError("crash after effect, before ack")

    await bus.publish({"k": "y"})
    await bus.consume_once("c1", handler, block_ms=50)
    await bus.reclaim_once("c2", handler, min_idle_ms=0)

    assert effects == ["y"]  # side effect ran exactly once despite redelivery
