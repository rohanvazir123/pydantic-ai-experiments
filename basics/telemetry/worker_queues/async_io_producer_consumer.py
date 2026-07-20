import asyncio

from dataclasses import dataclass
import time
from typing import Protocol

class Worker(Protocol):
    """A producer or consumer that runs as its own asyncio task."""
    async def run(self) -> None:
        """The worker's main loop."""
        raise NotImplementedError

    def cancel(self) -> None:
        """Cancel this worker. Called when the pool is shutting down."""
        raise NotImplementedError

    def is_done(self) -> bool:
        """Return True if the worker has finished. Used to decide if the pool can exit."""
        raise NotImplementedError

    def is_cancelled(self) -> bool:
        """Return True if the worker was cancelled. Used to decide if the pool can exit."""
        raise NotImplementedError

    def start(self) -> asyncio.Task[None]:
        """Create and schedule this worker's own asyncio task."""
        raise NotImplementedError


class Producer(Worker):
    def __init__(self, id: int, queue: asyncio.Queue[str]) -> None:
        self.id = f"p{id}"
        self.queue = queue
        self.task: asyncio.Task[None] | None = None
        print(f"Init producer {self.id}")

    def start(self) -> asyncio.Task[None]:
        print(f"producer {self.id} start")
        self.task = asyncio.create_task(self.add_jobs())  # schedules add_jobs() to start now, not lazily on first await
        return self.task

    def cancel(self) -> None:
        assert self.task
        self.task.cancel()

    async def add_jobs(self) -> None:
        try:
            i = 0
            while True:
                # Blocks cleanly until space is available. No polling.
                await self.queue.put(f"{time.monotonic():.2f}-item-{self.id}-{i}")
                print(f"Producer {self.id} added item {i}")
                i += 1
                # Not required for scheduling: the queue is bounded, so put()
                # already yields on its own once full. This just paces
                # production to a realistic rate -- good practice regardless.
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            print(f"Producer {self.id} intercepted cancel signal. Exiting.")
            raise  # Always propagate the cancellation up


class Consumer(Worker):
    def __init__(self, id: int, queue: asyncio.Queue[str]) -> None:
        self.id = f"c{id}"
        self.queue = queue
        self.task: asyncio.Task[None] | None = None
        print(f"Init consumer {self.id}")

    def start(self) -> asyncio.Task[None]:
        print(f"consumer {self.id} start")
        self.task = asyncio.create_task(self.run())  # schedules run() to start now, not lazily on first await
        return self.task

    def cancel(self) -> None:
        assert self.task
        self.task.cancel()

    async def run(self) -> None:
        while True:
            try:
                # Blocks cleanly until an item arrives.
                item = await self.queue.get()
            except asyncio.CancelledError:
                print(f"Consumer {self.id} forced to stop.")
                raise
            try:
                print(f"Consumer {self.id} processed: {item}")
            finally:
                # EVERY successful get() must be matched by a task_done(),
                # even if processing raises (or is cancelled mid-processing) --
                # otherwise queue.join() waits on this item forever.
                self.queue.task_done()


async def main() -> None:
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=10)

    producers = [Producer(id=i, queue=queue) for i in range(1)]
    consumers = [Consumer(id=i, queue=queue) for i in range(1)]

    producer_tasks, consumer_tasks = [], []

    # 1. Start tasks in background
    # Note: we don't await them here, we just start them and let them run in the background
    for p in producers:
        producer_tasks.append(p.start())

    for c in consumers:
        consumer_tasks.append(c.start())

    # 2. TODO: signal handlers not implemented yet. Main thread should register
    # SIGINT/SIGTERM handlers (loop.add_signal_handler) that trigger the clean
    # shutdown below, instead of letting KeyboardInterrupt hit mid-run.

    # 3. Kill the producers first
    print("\n--- Stopping Producers ---")
    for p in producers:
        p.cancel()

    # We must await them to ensure they are fully dead before moving on
    await asyncio.gather(*producer_tasks, return_exceptions=True)

    # 4. Drain the backlog
    print("Draining remaining queue items...")
    await queue.join()  # Blocks until every item in the queue is processed

    # 5. Kill consumers now that the queue is guaranteed empty
    print("--- Stopping Consumers ---")
    for c in consumers:
        c.cancel()

    await asyncio.gather(*consumer_tasks, return_exceptions=True)
    print("System fully offline.")


if __name__ == "__main__":
    asyncio.run(main())
