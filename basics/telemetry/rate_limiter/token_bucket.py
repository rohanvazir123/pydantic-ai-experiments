'''To implement a distributed Token Bucket rate limiter in Python and Redis without using Lua scripts, you must handle the classic "read-modify-write" race condition. Without Lua, the standard approach is to use a Redis transaction (MULTI/EXEC) with optimistic locking (WATCH). This ensures that if another application instance modifies the client's bucket while your code is calculating tokens, the transaction safely aborts and retries. [1, 2, 3, 4]
Core Implementation
The following complete, thread-safe, and distributed token bucket implementation 
utilizes redis-py with a WATCH block to handle concurrency safely:
'''

'''
Rate Limiter: Implement a thread-safe distributed rate limiter (e.g., token bucket) 
that can handle millions of incoming requests from ground contro
'''

import time
import redis

class RedisTokenBucketRateLimiter:
    def __init__(self, redis_client: redis.Redis, capacity: int, refill_rate: float):
        """
        :param redis_client: Initialized redis-py instance
        :param capacity: Maximum number of tokens the bucket can hold (burst size)
        :param refill_rate: Tokens added to the bucket per second
        """
        self.redis = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate

    def is_allowed(self, client_id: str, tokens_to_consume: int = 1) -> bool:
        # We store the state in a Redis Hash to fetch both fields in one call
        key = f"ratelimit:token_bucket:{client_id}"
        
        # Calculate max life of the key to automatically clean up memory
        max_idle_ttl = int((self.capacity / self.refill_rate) * 2) + 60

        while True:
            try:
                # 1. WATCH the key for any changes made by concurrent requests
                self.redis.watch(key)
                
                # 2. Fetch current bucket state
                state = self.redis.hgetall(key)
                now = time.time()

                if not state:
                    # First request: Initialize bucket to full capacity
                    current_tokens = float(self.capacity)
                    last_refill = now
                else:
                    # Cast byte strings from Redis back to appropriate types
                    last_tokens = float(state[b'tokens'])
                    last_refill = float(state[b'last_refill'])
                    
                    # 3. Calculate how many tokens accumulated since last check
                    elapsed = max(0.0, now - last_refill)
                    tokens_to_add = elapsed * self.refill_rate
                    current_tokens = min(self.capacity, last_tokens + tokens_to_add)

                # 4. Check if the bucket has enough tokens
                if current_tokens >= tokens_to_consume:
                    new_tokens = current_tokens - tokens_to_consume
                    is_allowed = True
                else:
                    new_tokens = current_tokens
                    is_allowed = False

                # 5. Open a MULTI transaction pipeline to save state atomically
                pipeline = self.redis.pipeline(transaction=True)
                
                pipeline.hset(key, mapping={
                    "tokens": new_tokens,
                    "last_refill": now
                })
                pipeline.expire(key, max_idle_ttl)
                
                # 6. Execute transaction. 
                # Raises WatchError if another process modified the key during step 2-5
                pipeline.execute()
                
                return is_allowed

            except redis.WatchError:
                # Concurrency clash detected! Retry the calculation loop safely.
                continue


Usage Example

# Initialize Redis connection
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

# Allow max 10 requests burst, refilling at 2 requests per second
limiter = RedisTokenBucketRateLimiter(redis_client=r, capacity=10, refill_rate=2.0)

# Simulate requests
user_id = "user_12345"

for i in range(12):
    if limiter.is_allowed(user_id):
        print(f"Request {i+1}: Allowed")
    else:
        print(f"Request {i+1}: Rate Limited (429)")
    time.sleep(0.1)
