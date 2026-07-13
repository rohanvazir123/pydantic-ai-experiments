"""Distributed token-bucket rate limiter backed by Redis (no Lua).

This is the reviewed/corrected version of ``token_bucket.py``. The original is
retained unchanged for comparison; this file holds the working implementation
plus full documentation. See ``test_token_bucket_fixed.py`` for the test suite.

Problem
-------
Implement a thread-safe, distributed rate limiter that many application
instances can share, capable of handling a very high request volume.

Why this is tricky
------------------
The token-bucket algorithm is a classic read-modify-write cycle:

    1. read the bucket's current token count,
    2. refill it based on elapsed time,
    3. decide whether to allow the request and write the new count back.

If two instances run those steps concurrently against the same key, one can
overwrite the other's update and let more requests through than the limit
allows. The usual fix is a Lua script (executed atomically server-side), but
this implementation deliberately avoids Lua and instead uses Redis optimistic
locking: WATCH the key, then run the write inside a MULTI/EXEC transaction. If
any other client modifies the key between the WATCH and the EXEC, Redis aborts
the transaction and raises ``WatchError``; we simply retry the whole cycle.

What was wrong with the original
--------------------------------
The original called ``self.redis.watch(key)`` on the pooled client but then ran
the transaction on a *separate* ``self.redis.pipeline()``. For WATCH to guard a
transaction, the WATCH, the reads, and the MULTI/EXEC must all run on the
*same* Redis connection. redis-py binds a connection to a ``Pipeline`` object
for exactly this purpose, so all of that work must go through one ``pipeline()``
instance. As originally written the optimistic lock silently did nothing (the
two calls used different pooled connections), ``WatchError`` never fired, and
the watched connection was never reset. This version routes ``watch``, the
read, and ``multi``/``execute`` through a single pipeline.
"""

import time

import redis


class RedisTokenBucketRateLimiter:
    """Token-bucket rate limiter whose state lives in a Redis hash.

    Each client gets one Redis hash holding two fields: ``tokens`` (how many
    tokens remain) and ``last_refill`` (the epoch time the bucket was last
    updated). Tokens are refilled lazily on each request based on the time
    elapsed since ``last_refill`` rather than by a background job, so the bucket
    costs nothing while idle.
    """

    def __init__(self, redis_client: redis.Redis, capacity: int, refill_rate: float) -> None:
        """Create a limiter.

        :param redis_client: Initialized redis-py client. It must be created
            with ``decode_responses=False`` (the default) because this class
            reads the hash fields as raw bytes.
        :param capacity: Maximum number of tokens the bucket can hold. This is
            the burst size — the most requests allowed in an instant.
        :param refill_rate: Tokens added back to the bucket per second. This is
            the sustained request rate once the initial burst is spent.
        """
        self.redis = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate

    def is_allowed(self, client_id: str, tokens_to_consume: int = 1) -> bool:
        """Return ``True`` if ``client_id`` may consume ``tokens_to_consume`` now.

        Consumes the tokens as a side effect when the request is allowed. When
        the bucket lacks enough tokens the request is denied and the bucket is
        left unchanged (aside from the lazy refill).

        The whole operation runs under Redis optimistic locking and retries on
        contention, so it is safe to call from many processes at once against
        the same ``client_id``.
        """
        # One hash per client. Storing both fields together lets us fetch the
        # entire bucket state in a single round trip.
        key = f"ratelimit:token_bucket:{client_id}"

        # Expire idle buckets so keys for one-off clients don't accumulate. A
        # full bucket refills in ``capacity / refill_rate`` seconds; we keep the
        # key around for twice that (plus a floor) so an active client's state
        # is never evicted mid-use, while abandoned keys are reclaimed.
        max_idle_ttl = int((self.capacity / self.refill_rate) * 2) + 60

        # A single pipeline object binds all commands below to one connection,
        # which is what makes WATCH actually guard the MULTI/EXEC.
        with self.redis.pipeline() as pipe:
            while True:
                try:
                    # 1. Start watching the key. From here until EXEC, any
                    #    concurrent write to this key will abort our transaction.
                    pipe.watch(key)

                    # 2. Read the current bucket state. While watching, the
                    #    pipeline runs commands immediately instead of queuing.
                    state = pipe.hgetall(key)
                    now = time.time()

                    if not state:
                        # First request ever for this client: start full.
                        current_tokens = float(self.capacity)
                    else:
                        # Redis returns bytes; cast back to the stored types.
                        last_tokens = float(state[b"tokens"])
                        last_refill = float(state[b"last_refill"])

                        # 3. Lazily refill: add the tokens that would have
                        #    accrued since the last update, capped at capacity.
                        elapsed = max(0.0, now - last_refill)
                        tokens_to_add = elapsed * self.refill_rate
                        current_tokens = min(self.capacity, last_tokens + tokens_to_add)

                    # 4. Decide. Only spend tokens if there are enough.
                    if current_tokens >= tokens_to_consume:
                        new_tokens = current_tokens - tokens_to_consume
                        allowed = True
                    else:
                        new_tokens = current_tokens
                        allowed = False

                    # 5. Switch the pipeline into buffered transaction mode; the
                    #    following commands are queued and run atomically at EXEC.
                    pipe.multi()
                    pipe.hset(key, mapping={"tokens": new_tokens, "last_refill": now})
                    pipe.expire(key, max_idle_ttl)

                    # 6. EXEC. Raises WatchError if the key changed since step 1,
                    #    which means our read is stale and we must recompute.
                    pipe.execute()

                    return allowed

                except redis.WatchError:
                    # Another instance updated the bucket first. Loop and retry
                    # with fresh state; execute() has already reset the pipeline.
                    continue


if __name__ == "__main__":
    # Demo: allow bursts of up to 10 requests, refilling at 2 tokens/second.
    # Requires a Redis server on localhost:6379.
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)
    limiter = RedisTokenBucketRateLimiter(redis_client=r, capacity=10, refill_rate=2.0)

    user_id = "user_12345"
    for i in range(12):
        if limiter.is_allowed(user_id):
            print(f"Request {i + 1}: Allowed")
        else:
            print(f"Request {i + 1}: Rate Limited (429)")
        time.sleep(0.1)
