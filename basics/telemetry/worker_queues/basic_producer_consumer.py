import threading
import time
import random
from queue import Queue

# Shared thread-safe buffer with a fixed max size
buffer = Queue(maxsize=5)

class Producer(threading.Thread):
    def run(self):
        while True:
            item = random.randint(1, 100)

            # block=True waits automatically if the queue is full
            buffer.put(item, block=True)
            print(f"Produced: {item}. Buffer size: {buffer.qsize()}")

            time.sleep(random.random())

class Consumer(threading.Thread):
    def run(self):
        while True:
            # block=True waits automatically if the queue is empty
            item = buffer.get(block=True)
            print(f"Consumed: {item}. Buffer size: {buffer.qsize()}")

            # Signals back to the queue that the task is complete
            buffer.task_done()

            time.sleep(random.random())

# Start the threads
Producer().start()
Consumer().start()
