"""Minimal at-least-once work queue on Redis Streams.

Three loops make up the whole delivery guarantee:

    produce  -> XADD
    consume  -> XREADGROUP ">"   then XACK on success
    reclaim  -> XAUTOCLAIM        (takes over messages a crashed consumer
                                   read but never ACKed — this is the piece a
                                   managed broker like SQS/Pub-Sub does for you)

Because delivery is *at-least-once*, handlers MUST be idempotent. A poison
message that keeps failing is routed to a dead-letter stream after
``max_attempts`` so it can't block the group forever.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import redis.asyncio as redis

log = logging.getLogger("bus")

# handler(msg_id, fields) -> None; raise to signal failure (message stays pending)
Handler = Callable[[str, dict[str, str]], Awaitable[None]]


@dataclass
class RedisStreamBus:
    client: redis.Redis
    stream: str
    group: str
    dlq_stream: str = ""
    max_attempts: int = 5

    def __post_init__(self) -> None:
        if not self.dlq_stream:
            self.dlq_stream = f"{self.stream}:dlq"

    async def ensure_group(self) -> None:
        """Create the consumer group (idempotent — BUSYGROUP means it exists)."""
        try:
            await self.client.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def publish(self, fields: dict[str, str]) -> str:
        """Enqueue one message; returns its stream id."""
        return await self.client.xadd(self.stream, fields)

    async def consume_once(
        self, consumer: str, handler: Handler, count: int = 10, block_ms: int = 1000
    ) -> int:
        """Process a batch of never-delivered messages ('>'). Returns count handled."""
        resp = await self.client.xreadgroup(
            self.group, consumer, {self.stream: ">"}, count=count, block=block_ms
        )
        handled = 0
        for _stream, messages in resp or []:
            for msg_id, fields in messages:
                await self._dispatch(msg_id, fields, handler, attempt=1)
                handled += 1
        return handled

    async def reclaim_once(
        self, consumer: str, handler: Handler, min_idle_ms: int = 30_000, count: int = 10
    ) -> int:
        """Take over up to ``count`` messages idle longer than ``min_idle_ms``.

        This is the pass that replaces a managed broker's automatic redelivery: a
        crashed consumer's messages sit unacked in its PEL until another consumer
        claims them here. One batch per call — the ``run`` loop repeats it, so a
        large backlog is drained across successive ticks (and we don't depend on
        the XAUTOCLAIM cursor sentinel, which differs between Redis and fakeredis).
        """
        _cursor, claimed, _deleted = await self.client.xautoclaim(
            self.stream,
            self.group,
            consumer,
            min_idle_time=min_idle_ms,
            start_id="0-0",
            count=count,
        )
        for msg_id, fields in claimed:
            attempt = await self._delivery_count(msg_id)
            await self._dispatch(msg_id, fields, handler, attempt=attempt)
        return len(claimed)

    async def _dispatch(
        self, msg_id: str, fields: dict[str, str], handler: Handler, attempt: int
    ) -> None:
        """Run the handler; ACK on success, DLQ after max_attempts, else leave pending."""
        try:
            await handler(msg_id, fields)
            await self.client.xack(self.stream, self.group, msg_id)
        except Exception:
            log.exception("handler failed id=%s attempt=%s", msg_id, attempt)
            if attempt >= self.max_attempts:
                await self.client.xadd(
                    self.dlq_stream,
                    {**fields, "_failed_id": msg_id, "_attempts": str(attempt)},
                )
                await self.client.xack(self.stream, self.group, msg_id)  # drop from PEL
                log.warning("id=%s -> DLQ after %s attempts", msg_id, attempt)
            # else: no ACK -> stays in the PEL -> reclaimed on a later pass

    async def _delivery_count(self, msg_id: str) -> int:
        """How many times this message has been delivered (from the PEL)."""
        pending = await self.client.xpending_range(
            self.stream, self.group, min=msg_id, max=msg_id, count=1
        )
        return int(pending[0]["times_delivered"]) if pending else 1

    async def run(self, consumer: str, handler: Handler, min_idle_ms: int = 30_000) -> None:
        """Production loop: interleave consume + reclaim forever."""
        await self.ensure_group()
        while True:  # pragma: no cover - long-running loop
            await self.consume_once(consumer, handler)
            await self.reclaim_once(consumer, handler, min_idle_ms=min_idle_ms)
