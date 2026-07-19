import asyncio


class Producer:
    def __init__(self, id: int, queue: asyncio.Queue[str]) -> None:
        self.id = f"p{id}"
        self.queue = queue
        self.task: asyncio.Task[None] | None = None
        print(f"Init producer {self.id}")

    def start(self) -> asyncio.Task[None]:
        print(f"producer {self.id} start")
        self.task = asyncio.create_task(self.run())  # schedules run() to start now, not lazily on first await
        return self.task

    def cancel(self) -> None:
        assert self.task
        self.task.cancel()

    async def run(self) -> None:
        try:
            i = 0
            while True:
                # Blocks cleanly until space is available. No polling.
                await self.queue.put(f"item-{self.id}-{i}")
                print(f"Producer {self.id} added item {i}")
                i += 1
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            print(f"Producer {self.id} intercepted cancel signal. Exiting.")
            raise  # Always propagate the cancellation up


class Consumer:
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
        try:
            while True:
                # Blocks cleanly until an item arrives.
                item = await self.queue.get()
                print(f"Consumer {self.id} processed: {item}")
                self.queue.task_done()
        except asyncio.CancelledError:
            print(f"Consumer {self.id} forced to stop.")
            raise


async def main() -> None:
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=10)

    producers = [Producer(id=i, queue=queue) for i in range(1)]
    consumers = [Consumer(id=i, queue=queue) for i in range(1)]

    producer_tasks, consumer_tasks = [], []

    # 1. Start tasks in background
    for p in producers:
        producer_tasks.append(p.start())

    for c in consumers:
        consumer_tasks.append(c.start())

    await asyncio.sleep(5)  # Let them run

    # 2. Kill the producers first
    print("\n--- Stopping Producers ---")
    for p in producers:
        p.cancel()

    # We must await them to ensure they are fully dead before moving on
    await asyncio.gather(*producer_tasks, return_exceptions=True)

    # 3. Drain the backlog
    print("Draining remaining queue items...")
    await queue.join()  # Blocks until every item in the queue is processed

    # 4. Kill consumers now that the queue is guaranteed empty
    print("--- Stopping Consumers ---")
    for c in consumers:
        c.cancel()

    # await
    await asyncio.gather(*consumer_tasks, return_exceptions=True)
    print("System fully offline.")


if __name__ == "__main__":
    asyncio.run(main())
