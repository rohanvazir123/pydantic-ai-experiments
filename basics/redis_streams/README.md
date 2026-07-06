# redis_streams — an at-least-once work queue on Redis Streams

A tiny, runnable reference for the **consumer-group + reclaim + idempotency + DLQ**
pattern — the thing a managed broker (SQS, Pub/Sub) gives you for free, built by
hand so you can see exactly what "you own redelivery" means.

## The whole idea in three loops

| Loop | Command | Who does it |
|------|---------|-------------|
| **produce** | `XADD` | producer |
| **consume new** | `XREADGROUP … >` then `XACK` on success | each worker |
| **reclaim stuck** | `XAUTOCLAIM` (idle > threshold) | each worker, periodically |

A message read by a worker sits in that worker's **PEL** (pending list) until it's
`XACK`ed. If the worker crashes, Redis does **not** reassign it — another worker must
`XAUTOCLAIM` it. That reclaim loop is the piece SQS/Pub-Sub do invisibly via an ack
deadline / visibility timeout.

Because delivery is **at-least-once**, handlers must be **idempotent** (here: a Redis
`SET … NX` dedup guard). A message that keeps failing is routed to a **dead-letter
stream** after `max_attempts` so it can't wedge the group.

## Files

- `bus.py` — `RedisStreamBus`: `publish` / `consume_once` / `reclaim_once` / `run` + DLQ.
- `demo.py` — proves redelivery: a consumer "crashes" on one message, a reclaimer
  picks it up, and the side effect still runs **exactly once**.
- `test_bus.py` — happy path, reclaim-after-failure, DLQ, and idempotency, on fakeredis.

## Run it

```bash
cd basics/redis_streams
uv sync

# tests — no services needed (fakeredis)
uv run pytest

# the crash/redelivery proof — fakeredis by default (zero infra)
uv run python demo.py

# ...or against real Redis
docker run -d --rm -p 6379:6379 redis
REDIS_URL=redis://localhost:6379 uv run python demo.py
```

Expected tail of `demo.py`:

```
PASS ✓  order 2 was redelivered after the 'crash' and its side effect ran exactly once.
```

## Talking points (why this, and the honest trade-off)

- **The real choice is who owns redelivery.** A managed broker (SQS visibility timeout,
  Pub/Sub ack deadline) redelivers automatically; Redis Streams pushes that to you as the
  `XAUTOCLAIM` reclaim loop. More control, more ops. Default to the managed broker unless
  you already run Redis or need its low-latency in-memory characteristics.
- **The invariant that transfers to every broker:** at-least-once ⇒ idempotent consumers.
  The dedup guard is a fast path; for correctness, push idempotency to the *destination*
  (e.g. a payment API keyed by an idempotency token) so a crash between effect and ACK
  can't double-charge.
- **Ordering:** per-key order comes from partitioning (one stream per key, or a hash of
  keys to N streams) — same idea as an SQS FIFO `MessageGroupId` or a Pub/Sub ordering key.
- **AWS mapping:** `SQS` ≈ this managed; `SNS→SQS` ≈ fan-out; `Kinesis/MSK` ≈ the
  log/offset model (consumer owns position, replayable), which is closer to Redis Streams
  than to SQS.
