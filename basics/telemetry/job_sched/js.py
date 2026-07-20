
"""
Priority Queue Application:
Implement a scheduling system that executes tasks based on safety priority levels
and timestamp  constraints

"""

"""
An efficient task scheduling system uses a Priority Queue built with a Min-Heap.

Tuple Evaluation Order:
The __lt__ (less than) method overrides default comparisons.
The heap evaluates priority first. If equal, it evaluates timestamp.

Timestamp Constraints:
The execute_next function peeks at the top element using self._queue[0].
It verifies if the system time meets or exceeds the required timestamp before removing it
from the heap.

Time Complexity:
Inserting a task (heappush) runs in O(log N) time.
Fetching and executing the next highest priority task (heappop) also runs in O(log N) time
"""


import time
from dataclasses import dataclass, field
import heapq
from typing import Any
import asyncio
from collections import deque
from typing import Protocol

@dataclass(order=False)
class Task:
    priority: int       # Lower numbers = Higher priority (e.g., 1 = Critical, 2 = Warning)
    timestamp: float    # Unix timestamp for tie-breaking FIFO order
    name: str          # Name/Identifier of the task
    payload: Any = field(default=None, repr=False) # Extra data or metadata

    # Define strict comparison for the min-heap
    def __lt__(self, other: 'Task') -> bool:
        if self.priority == other.priority:
            return self.timestamp < other.timestamp
        return self.priority < other.priority


class TaskScheduler(Protocol):
    def schedule_task(self, name: str, priority: int, delay_seconds: float = 0.0) -> None:
        """Adds a task to the queue with a priority and future timestamp."""
        raise NotImplementedError

    def execute_next(self) -> None:
        """Executes the highest priority, valid task ready for execution."""
        raise NotImplementedError


class SafetyTaskSchedulerUsingHeap(TaskScheduler):
    def __init__(self):

        # Unbounded min-heap priority queue for tasks, with lower numbers indicating higher priority
        self._queue: list[Task] = []

    def schedule_task(self, name: str, priority: int, delay_seconds: float = 0.0) -> None:
        """Adds a task to the queue with a priority and future timestamp."""
        # Calculate execution timestamp
        execution_time = time.monotonic() + delay_seconds
        task = Task(priority=priority, timestamp=execution_time, name=name)
        heapq.heappush(self._queue, task)
        print(f"[Scheduled] '{name}' | Priority: {priority} | Execution Time: {execution_time:.2f}")

    def execute_next(self) -> None:
        """Executes the highest priority, valid task ready for execution."""
        if not self._queue:
            print("[Empty] No tasks left to execute.")
            return

        # Peek at the root element without popping
        next_task = self._queue[0]

        # monotonic time is preferred for measuring elapsed time, as it is not affected by system clock updates
        current_time = time.monotonic()

        # Enforce timestamp constraint (cannot run before its time)
        if current_time < next_task.timestamp:
            wait_time = next_task.timestamp - current_time
            print(f"[Blocked] Next task '{next_task.name}' must wait {wait_time:.2f} seconds.")
            return

        # Safely pop and execute
        task = heapq.heappop(self._queue)
        print(f"[Executing] '{task.name}' (Priority {task.priority}) at timestamp {current_time:.2f}")

    def is_empty(self) -> bool:
        return len(self._queue) == 0

class SafetyTaskSchedulerUsingAsyncHeap(TaskScheduler):
    def __init__(self):

        # min-heap priority queue for tasks, with lower numbers indicating higher priority
        # unbounded queue, but we will enforce timestamp constraints in execute_next
        self._queue: asyncio.PriorityQueue[Task] = asyncio.PriorityQueue()

    def schedule_task(self, name: str, priority: int, delay_seconds: float = 0.0) -> None:
        """Adds a task to the queue with a priority and future timestamp."""
        execution_time = time.monotonic() + delay_seconds
        task = Task(priority=priority, timestamp=execution_time, name=name)

        # Use put_nowait to avoid blocking in the async context
        self._queue.put_nowait(task)
        print(f"[Scheduled] '{name}' | Priority: {priority} | Execution Time: {execution_time:.2f}")

    def execute_next(self) -> None:
        """Executes the highest priority, valid task ready for execution."""
        if self._queue.empty():
            print("[Empty] No tasks left to execute.")
            return

        # Peek at the root element without popping
        next_task = self._queue._queue[0]  # Accessing the underlying deque for peeking
        current_time = time.monotonic()

        # Enforce timestamp constraint (cannot run before its time)
        if current_time < next_task.timestamp:
            wait_time = next_task.timestamp - current_time
            print(f"[Blocked] Next task '{next_task.name}' must wait {wait_time:.2f} seconds.")
            return

        # Safely pop and execute
        task = self._queue.get_nowait()
        print(f"[Executing] '{task.name}' (Priority {task.priority}) at timestamp {current_time:.2f}")

def test_safety_task_scheduler_using_async_heap():

    # Initialize the scheduler
    scheduler = SafetyTaskSchedulerUsingAsyncHeap()

    # 1. Enqueue tasks with varying priorities and timestamps
    print("--- 1. Enqueueing Tasks ---")
    scheduler.schedule_task("Routine Log Cleanup", priority=3)
    scheduler.schedule_task("CRITICAL: Core Temp Warning", priority=1)
    scheduler.schedule_task("Delayed System Check", priority=1, delay_seconds=2.0)
    scheduler.schedule_task("Brake Failure Alert", priority=1)

    #  2. Execute tasks in order of priority and timestamp
    print("\n--- 2. Processing Tasks ---")
    scheduler.execute_next()
    scheduler.execute_next()
    scheduler.execute_next()

    # 2.1 seconds sleep to allow delayed task to become eligible
    print("\n[Sleeping 2.1 seconds...]")
    time.sleep(2.1)

    # 3. Execute remaining tasks after delay
    scheduler.execute_next()
    scheduler.execute_next()

def test_safety_task_scheduler_using_heap():
    # Initialize the scheduler
    scheduler = SafetyTaskSchedulerUsingHeap()

    # 1. Enqueue tasks with varying priorities and timestamps
    print("--- 1. Enqueueing Tasks ---")
    scheduler.schedule_task("Routine Log Cleanup", priority=3)
    scheduler.schedule_task("CRITICAL: Core Temp Warning", priority=1)
    scheduler.schedule_task("Delayed System Check", priority=1, delay_seconds=2.0)
    scheduler.schedule_task("Brake Failure Alert", priority=1)

    # 2. Execute tasks in order of priority and timestamp
    print("\n--- 2. Processing Tasks ---")
    scheduler.execute_next()
    scheduler.execute_next()
    scheduler.execute_next()

    # 2.1 seconds sleep to allow delayed task to become eligible
    print("\n[Sleeping 2.1 seconds...]")
    time.sleep(2.1)

    # 3. Execute remaining tasks after delay
    scheduler.execute_next()
    scheduler.execute_next()

# --- Example System Execution ---
if __name__ == "__main__":
    print("\n--- Testing SafetyTaskSchedulerUsingHeap ---")
    test_safety_task_scheduler_using_heap()

    print("\n--- Testing SafetyTaskSchedulerUsingAsyncHeap ---")
    test_safety_task_scheduler_using_async_heap()
