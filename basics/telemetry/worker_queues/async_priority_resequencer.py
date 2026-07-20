import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

# Same trick as async_priority_queue.py's min-heap, minus a private per-device
# buffer: the priority queue IS the buffer. A frame that isn't next just goes
# back in; it keeps losing to whatever's smaller until the gap-filler lands,
# or until `timeout` gives up on it -- the gap is logged to gap_queue and the
# frame is processed anyway so nothing behind it stays stuck forever.
# Assumes idempotent, non-duplicate delivery -- a duplicate of an already
# emitted seq would never match `expected` again and spins forever.

@dataclass(order=False)
class Frame:
    seq: int            # position this frame must land at in the output
    payload: Any = field(default=None, repr=False)

    # Define strict comparison for the min-heap
    def __lt__(self, other: 'Frame') -> bool:
        return self.seq < other.seq

async def resequencer(
    in_queue: asyncio.PriorityQueue[Frame],
    out_queue: asyncio.Queue[Frame],
    gap_queue: asyncio.Queue[tuple[int, int]],
    timeout: float,
):
    expected = 0
    blocked_since: float | None = None
    while True:
        frame = await in_queue.get()
        now = time.monotonic()
        if frame.seq == expected:
            await out_queue.put(frame)
            expected += 1
            blocked_since = None
        elif blocked_since is not None and now - blocked_since > timeout:
            await gap_queue.put((expected, frame.seq))  # gave up -- note the gap
            await out_queue.put(frame)                  # and process it anyway
            expected = frame.seq + 1
            blocked_since = None
        else:
            blocked_since = blocked_since or now  # first time we've seen this gap
            await in_queue.put(frame)              # not next yet -- back on the queue
            await asyncio.sleep(0)                 # yield, or the producer never gets scheduled
        in_queue.task_done()

async def producer(queue: asyncio.PriorityQueue[Frame], n: int, skip: int | None = None):
    order = [seq for seq in range(n) if seq != skip]
    random.shuffle(order)  # simulate frames completing out of order
    for seq in order:
        await asyncio.sleep(random.uniform(0.001, 0.01))
        await queue.put(Frame(seq=seq, payload=f"frame-{seq}"))

async def main():
    in_queue: asyncio.PriorityQueue[Frame] = asyncio.PriorityQueue()
    out_queue: asyncio.Queue[Frame] = asyncio.Queue()
    gap_queue: asyncio.Queue[tuple[int, int]] = asyncio.Queue()

    producer_task = asyncio.create_task(producer(in_queue, 10, skip=5))  # seq=5 never arrives
    resequencer_task = asyncio.create_task(resequencer(in_queue, out_queue, gap_queue, timeout=0.05))

    await producer_task
    await in_queue.join()
    resequencer_task.cancel()

    print("emitted order:", [out_queue.get_nowait().seq for _ in range(out_queue.qsize())])
    print("gaps:", [gap_queue.get_nowait() for _ in range(gap_queue.qsize())])

if __name__ == "__main__":
    asyncio.run(main())
