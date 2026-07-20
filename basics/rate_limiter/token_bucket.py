"""Distributed token-bucket rate limiter backed by Redis (no Lua).

Uses optimistic locking instead of a Lua script: WATCH the key, read it,
compute the new state, then write inside MULTI/EXEC. If another client
touches the key in between, EXEC raises WatchError and we retry.
"""

import time
from collections.abc import Callable

import redis


class RedisTokenBucketRateLimiter:
    """Token-bucket rate limiter whose state lives in a Redis hash.

    Each client gets one hash with ``tokens`` and ``last_refill``. Tokens are
    refilled lazily on each request based on elapsed time, so an idle bucket
    costs nothing.
    """

    def __init__(self, redis_client: redis.Redis, capacity: int, refill_rate: float) -> None:
        """redis_client must use decode_responses=False (hash fields are read as bytes)."""
        self.redis = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate
        # Keep an active client's bucket alive for two full refill cycles;
        # abandoned buckets expire instead of accumulating forever.
        self.max_idle_ttl = int((capacity / refill_rate) * 2) + 60

    def is_allowed(self, client_id: str, tokens_to_consume: int = 1) -> bool:
        """Return True and consume tokens if client_id has enough, else False."""

        def compute(state: dict[bytes, bytes]) -> tuple[bool, dict[str, float]]:
            now = time.time()

            if not state:
                current_tokens = float(self.capacity)
            else:
                last_tokens = float(state[b"tokens"])
                last_refill = float(state[b"last_refill"])
                elapsed = max(0.0, now - last_refill)
                current_tokens = min(self.capacity, last_tokens + elapsed * self.refill_rate)

            allowed = current_tokens >= tokens_to_consume
            new_tokens = current_tokens - tokens_to_consume if allowed else current_tokens
            return allowed, {"tokens": new_tokens, "last_refill": now}

        key = f"ratelimit:token_bucket:{client_id}"
        return self._watch_and_update(key, compute)

    def _watch_and_update[T](
        self,
        key: str,
        compute: Callable[[dict[bytes, bytes]], tuple[T, dict[str, float]]],
    ) -> T:
        """Run a WATCH/read/MULTI-EXEC cycle against a hash key, retrying on WatchError.

        `compute` receives the current hash state and returns the caller's result
        plus the fields to write back. WATCH only guards the transaction if the
        watch, the read, and the MULTI/EXEC all run on the same connection, so
        everything here goes through one `pipeline()` instance.
        """
        with self.redis.pipeline() as pipe:
            while True:
                try:

                    # read
                    pipe.watch(key)
                    state = pipe.hgetall(key)

                    # compute
                    result, fields = compute(state)

                    # write
                    pipe.multi()
                    pipe.hset(key, mapping=fields)
                    pipe.expire(key, self.max_idle_ttl)
                    pipe.execute()

                    return result

                except redis.WatchError:
                    continue


if __name__ == "__main__":
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)
    limiter = RedisTokenBucketRateLimiter(redis_client=r, capacity=10, refill_rate=2.0)

    user_id = "user_12345"
    for i in range(12):
        if limiter.is_allowed(user_id):
            print(f"Request {i + 1}: Allowed")
        else:
            print(f"Request {i + 1}: Rate Limited (429)")
        time.sleep(0.1)
