"""Bug-free producer/consumer using one mutex + two condition variables.

Two CVs over ONE shared lock: `consumer_cond` (producers wait on it) and `producer_cond`
(consumers wait on it). Each CV wakes exactly one *right-role* waiter, so we use
precise `notify()` — never `notify_all()` for normal handoffs. A consumer freeing a
slot signals `consumer_cond` (can only wake a producer); a producer adding an item
signals `producer_cond` (can only wake a consumer). The wrong-role-wakeup hazard that
forces broadcast on a single shared CV is structurally impossible here.

Both producer and consumer loop forever — there's no in-band shutdown item. Instead
the main thread cancels them with a `threading.Event`: it sets the event, then
`notify_all()`s both CVs once (the one broadcast exception) to wake any thread
currently blocked in `wait_for`, which re-checks its predicate, sees the event set,
and returns.

Correctness rules that keep this race- and deadlock-free:
  * Both CVs share ONE lock, so a single critical section guards the whole queue —
    the two waits can't interleave against inconsistent state.
  * All access to shared state (`buffer`) happens while holding that lock.
  * `wait_for` re-checks its predicate after the lock is re-acquired — guards against
    spurious wakeups and stale state.
  * The stop check is part of every wait predicate, so cancellation can never be
    missed by a thread already blocked waiting for space/items.

The pthreads equivalent of the producer's critical section, for reference:

    pthread_mutex_lock(&m);
    while (count == CAP && !stop) pthread_cond_wait(&consumer_cond, &m);
    if (!stop) { buf[tail++ % CAP] = item; count++; }
    pthread_cond_signal(&producer_cond);
    pthread_mutex_unlock(&m);

Both CVs take the same `&m`, exactly as both `Condition`s share `lock` below.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque

BUFFER_SIZE = 5


class Producer(threading.Thread):
    def __init__(
        self,
        id: str,
        buffer: deque[object],
        consumer_cond: threading.Condition,
        producer_cond: threading.Condition,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=id)
        self.id = id
        self._buffer = buffer
        self._consumer_cond = consumer_cond
        self._producer_cond = producer_cond
        self._stop_event = stop_event

    def _put(self, item: object) -> None:
        with self._producer_cond:
            self._producer_cond.wait_for(
                lambda: len(self._buffer) < BUFFER_SIZE or self._stop_event.is_set()
            )
            if self._stop_event.is_set():
                return
            self._buffer.append(item)
            print(f"Produced: {item}. Buffer size: {len(self._buffer)}")
            self._consumer_cond.notify()

    def run(self) -> None:
        i = 0
        while True:
            if self._stop_event.is_set():
                break
            self._put(f"item-{i}")
            i += 1
            time.sleep(random.random() * 0.1)


class Consumer(threading.Thread):
    def __init__(
        self,
        id: str,
        buffer: deque[object],
        consumer_cond: threading.Condition,
        producer_cond: threading.Condition,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=id)
        self.id = id
        self._buffer = buffer
        self._consumer_cond = consumer_cond
        self._producer_cond = producer_cond
        self._stop_event = stop_event

    def _get(self) -> object | None:
        with self._consumer_cond:
            self._consumer_cond.wait_for(
                lambda: len(self._buffer) > 0 or self._stop_event.is_set()
            )
            if not self._buffer:
                return None  # stopped with nothing left to drain
            item = self._buffer.popleft()
            print(f"Consumed: {item}. Buffer size: {len(self._buffer)}")
            self._producer_cond.notify()
            return item

    def run(self) -> None:
        while True:
            if self._stop_event.is_set():
                break
            self._get()


def main() -> None:
    buffer: deque[object] = deque()
    lock = threading.Lock()
    consumer_cond = threading.Condition(lock)   # producers wait here for space
    producer_cond = threading.Condition(lock)  # consumers wait here for an item
    stop_event = threading.Event()

    producer = Producer(
        id="p1", buffer=buffer, consumer_cond=consumer_cond,
        producer_cond=producer_cond, stop_event=stop_event,
    )
    consumer = Consumer(
        id="c1", buffer=buffer, consumer_cond=consumer_cond,
        producer_cond=producer_cond, stop_event=stop_event,
    )

    consumer.start()
    producer.start()

    try:
        time.sleep(2)  # let them run for a bit
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        with lock:
            producer_cond.notify_all()  # wake a consumer blocked in wait_for
            consumer_cond.notify_all()  # wake a producer blocked in wait_for

    producer.join()
    consumer.join()
    print("Done — cancelled by main thread.")


if __name__ == "__main__":
    main()
