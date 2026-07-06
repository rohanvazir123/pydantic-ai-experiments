"""Prove at-least-once redelivery + idempotency, end to end.

Story:
  1. Publish 3 orders; one is a "poison-once" message that fails the first time.
  2. Consumer A processes the batch: acks the good ones, FAILS on the poison
     message (simulating a crash mid-handling) -> it stays in A's PEL, unacked.
  3. Reclaimer B runs XAUTOCLAIM with min_idle_ms=0 -> takes over the stuck
     message; this time the handler succeeds and acks it.
  4. Assert every order's side effect ran EXACTLY ONCE (idempotency held) and the
     poison message was retried, not lost.

Runs on fakeredis by default (zero infra). Point at real Redis with:
    REDIS_URL=redis://localhost:6379 uv run python demo.py
"""

from __future__ import annotations

import asyncio
import os

from bus import RedisStreamBus


def make_client():
    url = os.environ.get("REDIS_URL")
    if url:
        import redis.asyncio as redis

        print(f"• using real Redis at {url}")
        return redis.from_url(url, decode_responses=True)
    import fakeredis

    print("• using fakeredis (in-process; set REDIS_URL to use real Redis)")
    return fakeredis.FakeAsyncRedis(decode_responses=True)


async def main() -> None:
    client = make_client()
    # fresh state so the demo is repeatable
    await client.delete("orders", "orders:dlq", *[f"done:{i}" for i in (1, 2, 3)])

    bus = RedisStreamBus(client=client, stream="orders", group="workers")
    await bus.ensure_group()

    # ---- observable state the handler mutates ----
    effects: dict[str, int] = {}   # order_id -> number of times the side effect ran
    attempts: dict[str, int] = {}  # order_id -> number of times the handler was entered

    async def handler(msg_id: str, fields: dict[str, str]) -> None:
        oid = fields["order_id"]
        attempts[oid] = attempts.get(oid, 0) + 1
        # simulate a crash the FIRST time we touch the poison message
        if fields.get("poison") == "1" and attempts[oid] == 1:
            raise RuntimeError(f"simulated crash handling order {oid}")
        # idempotent side effect: SETNX guard => runs at most once per order
        if await client.set(f"done:{oid}", "1", nx=True, ex=3600):
            effects[oid] = effects.get(oid, 0) + 1

    # 1. publish
    await bus.publish({"order_id": "1"})
    await bus.publish({"order_id": "2", "poison": "1"})
    await bus.publish({"order_id": "3"})

    # 2. consumer A: order 2 fails and is left pending
    handled = await bus.consume_once("A", handler, block_ms=100)
    print(f"\nconsumer A handled {handled} messages; attempts so far: {attempts}")

    pending = await client.xpending("orders", "workers")
    print(f"pending after A (should be 1 — the poison message): {pending['pending']}")

    # 3. reclaimer B: min_idle_ms=0 -> immediately claim the stuck message
    reclaimed = await bus.reclaim_once("B", handler, min_idle_ms=0)
    print(f"reclaimer B reclaimed {reclaimed} message(s); attempts now: {attempts}")

    pending = await client.xpending("orders", "workers")
    print(f"pending after B (should be 0): {pending['pending']}")

    # 4. assertions
    assert effects == {"1": 1, "2": 1, "3": 1}, f"exactly-once side effects broken: {effects}"
    assert attempts["2"] == 2, f"poison message should be tried twice, got {attempts['2']}"
    assert pending["pending"] == 0, "PEL should be drained"

    print("\nPASS ✓  order 2 was redelivered after the 'crash' and its side "
          "effect ran exactly once.")
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
