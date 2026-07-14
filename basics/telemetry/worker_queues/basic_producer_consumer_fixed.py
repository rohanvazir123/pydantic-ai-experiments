"""Bug-free producer/consumer using one mutex + two condition variables.

Two CVs over ONE shared lock: `not_full` (producers wait on it) and `not_empty`
(consumers wait on it). Each CV wakes exactly one *right-role* waiter, so we use
precise `notify()` — never `notify_all()`. A consumer freeing a slot signals
`not_full` (can only wake a producer); a producer adding an item signals
`not_empty` (can only wake a consumer). The wrong-role-wakeup hazard that forces
broadcast on a single shared CV is structurally impossible here.

Shutdown is signalled with a *sentinel*: the producer sends a poison-pill object
through the buffer as its final item; the consumer stops when it pulls it. The
sentinel travels in-band and preserves order, so everything queued before it is
drained first.

Correctness rules that keep this race- and deadlock-free:
  * Both CVs share ONE lock, so a single critical section guards the whole queue —
    the two waits can't interleave against inconsistent state.
  * All access to shared state (`buffer`) happens while holding that lock.
  * Waits sit in `while <predicate>:` loops, never `if` — guards against spurious
    wakeups and re-checks the predicate after the lock is re-acquired.
  * One sentinel stops one consumer. With N consumers, send N sentinels.

The pthreads equivalent of the producer's critical section, for reference:

    pthread_mutex_lock(&m);
    while (count == CAP) pthread_cond_wait(&not_full, &m);   // while, not if
    buf[tail++ % CAP] = item; count++;
    pthread_cond_signal(&not_empty);                         // precise, one waiter
    pthread_mutex_unlock(&m);

Both CVs take the same `&m`, exactly as both `Condition`s share `lock` below.
"""

from __future__ import annotations

import random
import threading
import time

BUFFER_SIZE = 5

# Unique poison-pill: distinct from any real payload, so identity checks are exact.
SENTINEL = object()

buffer: list[object] = []
lock = threading.Lock()
not_full = threading.Condition(lock)   # producers wait here for space
not_empty = threading.Condition(lock)  # consumers wait here for an item


class Producer(threading.Thread):
    def __init__(self, count: int) -> None:
        super().__init__(name="producer")
        self._count = count

    def _put(self, item: object) -> None:
        with not_full:
            while len(buffer) == BUFFER_SIZE:
                print("Buffer full. Producer is waiting...")
                not_full.wait()
            buffer.append(item)
            if item is not SENTINEL:
                print(f"Produced: {item}. Buffer size: {len(buffer)}")
            not_empty.notify()

    def run(self) -> None:
        for _ in range(self._count):
            self._put(random.randint(1, 100))
            time.sleep(random.random() * 0.1)
        self._put(SENTINEL)  # in-band shutdown signal, drained after everything else


class Consumer(threading.Thread):
    def run(self) -> None:
        while True:
            with not_empty:
                while not buffer:
                    print("Buffer empty. Consumer is waiting...")
                    not_empty.wait()
                item = buffer.pop(0)
                if item is SENTINEL:
                    return  # poison pill — no more items will arrive
                print(f"Consumed: {item}. Buffer size: {len(buffer)}")
                not_full.notify()


def main() -> None:
    producer = Producer(count=20)
    consumer = Consumer()

    consumer.start()
    producer.start()

    producer.join()
    consumer.join()
    print("Done — no deadlock, all items drained.")


if __name__ == "__main__":
    main()
