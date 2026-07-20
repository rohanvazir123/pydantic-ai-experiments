import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=False)
class Task:
    priority: int       # Lower numbers = Higher priority (e.g., 1 = Critical, 2 = Warning)
    timestamp: float    # Unix timestamp for tie-breaking FIFO order
    name: str           # Name/Identifier of the task
    payload: Any = field(default=None, repr=False)

    # Define strict comparison for the min-heap
    def __lt__(self, other: 'Task') -> bool:
        if self.priority == other.priority:
            return self.timestamp < other.timestamp
        return self.priority < other.priority

async def worker(name: str, queue: asyncio.PriorityQueue[Task]):
    while True:
        task = await queue.get()
        print(f"[{name}] Executing: {task.name} (Priority: {task.priority})")

        # Simulate background processing
        await asyncio.sleep(0.5)
        queue.task_done()

async def main():
    queue: asyncio.PriorityQueue[Task] = asyncio.PriorityQueue()

    # Enqueue a baseline low-priority item
    await queue.put(Task(priority=3, timestamp=time.time(), name="Backup Logs"))

    # Enqueue two matching high-priority items with a tiny time delay
    # Task A should execute BEFORE Task B because its timestamp is older
    t_start = time.monotonic()
    await queue.put(Task(priority=1, timestamp=t_start, name="Alert Admin A"))
    await queue.put(Task(priority=1, timestamp=t_start + 0.001, name="Alert Admin B"))

    # Enqueue a medium-priority item
    await queue.put(Task(priority=2, timestamp=time.time(), name="Generate Report"))

    # Spin up consumer worker
    consumer = asyncio.create_task(worker("Worker-1", queue))

    # Process all elements
    await queue.join()
    consumer.cancel()

if __name__ == "__main__":
    asyncio.run(main())
