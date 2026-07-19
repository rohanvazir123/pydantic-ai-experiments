import asyncio
import random

async def event_worker(worker_id: int, queue: asyncio.Queue):
    """This background worker runs forever processing stream events."""
    print(f"[Worker {worker_id}] Started and listening for events...")
    while True:
        try:
            event = await queue.get()
            print(f"[Worker {worker_id}] Processing: {event}")
            
            # Simulate real network/DB delay
            await asyncio.sleep(random.uniform(0.5, 1.5)) 
            
            print(f"[Worker {worker_id}] Finished: {event}")
        except Exception as e:
            print(f"[Worker {worker_id}] Error processing event: {e}.")
        finally:
            if 'event' in locals():
                queue.task_done()

async def simulate_incoming_traffic(queue: asyncio.Queue):
    """Simulates an infinite stream of real-time data arriving over time."""
    event_id = 1
    while True:
        await queue.put(f"RealTime-Event-{event_id}")
        event_id += 1
        # Wait 2 seconds before the next event arrives
        await asyncio.sleep(2.0)

async def main():
    event_queue = asyncio.Queue()
    worker_tasks = set()
    num_workers = 3
    
    # 1. Start the permanent background workers
    for i in range(num_workers):
        task = asyncio.create_task(event_worker(worker_id=i+1, queue=event_queue))
        worker_tasks.add(task) # Kept safe from garbage collection
        
    # 2. Start your data source (e.g. your real Kafka/RabbitMQ consumer feed)
    asyncio.create_task(simulate_incoming_traffic(event_queue))
    
    print("Main: System is fully operational. Running forever...")
    
    # 3. Keep the main function alive indefinitely
    # Because we never cancel the tasks and never exit main(), they run forever.
    while True:
        await asyncio.sleep(3600)  # Sleep for an hour, looping infinitely

asyncio.run(main())
