# Redis Token-Bucket Rate Limiter

A distributed, thread-safe token-bucket rate limiter built on Redis **without
Lua**, using optimistic locking (`WATCH` / `MULTI` / `EXEC`) to make the
read-modify-write cycle safe across many application instances.

## Table of Contents

- [Files](#files)
- [How It Works](#how-it-works)
- [Running the Demo](#running-the-demo)
- [Running the Tests](#running-the-tests)

## Files

| File | Purpose |
|------|---------|
| `token_bucket.py` | Original draft (kept for reference; do not build on it). |
| `token_bucket_fixed.py` | Reviewed, corrected, and documented implementation. |
| `test_token_bucket_fixed.py` | Pytest suite (uses `fakeredis`, no server needed). |

### Why two files

`token_bucket.py` had two problems: it didn't parse (a "Usage Example" block sat
outside its docstring), and its optimistic lock was ineffective — it called
`WATCH` on the pooled client but ran the transaction on a *separate* pipeline
(different connection), so the lock did nothing and `WatchError` never fired.
`token_bucket_fixed.py` routes `watch`, the read, and `multi`/`execute` through
a **single** pipeline (one connection), which is what makes `WATCH` actually
guard the transaction.

## How It Works

Each client's bucket is a Redis hash with two fields: `tokens` (remaining) and
`last_refill` (epoch seconds of the last update). On each request the limiter:

1. `WATCH`es the client's key on a pipeline-bound connection.
2. Reads the current state and **lazily refills** tokens based on elapsed time
   (no background job), capped at `capacity`.
3. Allows the request if enough tokens remain, decrementing them.
4. Persists the new state inside a `MULTI`/`EXEC` transaction with a TTL so idle
   buckets self-clean.
5. On `WatchError` (a concurrent writer touched the key), retries the whole
   cycle against fresh state.

`capacity` is the burst size; `refill_rate` is the sustained requests/second.

## Running the Demo

Requires a Redis server on `localhost:6379`:

```bash
python token_bucket_fixed.py
```

## Running the Tests

No Redis server required — the suite uses `fakeredis` and a monkeypatched clock,
so refill/expiry behavior is deterministic. From the repo root:

```bash
.venv/bin/python -m pytest basics/telemetry/tests/test_token_bucket_fixed.py -v
```

Dependencies: `redis`, `fakeredis`, `pytest`.
