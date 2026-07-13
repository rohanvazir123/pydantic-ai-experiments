# =====================================================================
# 1. IO-BOUND (asyncio: Telemetry APIs, Webhooks, DB Logs)
# =====================================================================
import asyncio
from dbm import sqlite3
from pyclbr import Class
import queue

from pydantic import BaseModel

import sqlite3

class TelemetryData(BaseModel):
    device_id: str
    metric: dict

class IoWorkerQueue:
    def __init__(self, maxsize=100, num_workers=5):
        self.io_queue = asyncio.Queue(maxsize)  # Limit the queue size to 10
    
        # 2. Spin up a fixed-size worker pool
        self.workers = [
            asyncio.create_task(self.process_io_work(i)) 
            for i in range(num_workers)
        ]

        # Are multiple workers neeeded? For now, we are using a single worker for simplicity.
        #     # If needed, we can create multiple workers by starting multiple tasks here.

    def insert_io_task(self, telemetry_data: TelemetryData):
        # Direct queue insertion. Blocks thread if queue maxsize is reached.
        print(f"Inserting telemetry data for device {telemetry_data.device_id} into IO queue.")
        self.io_queue.put_nowait(telemetry_data)


    # The Consumer Pool Loop
    async def process_io_work(self, worker_id: int):
        print(f"IO Worker {worker_id} started processing IO tasks.")
        while True:
            io_work = await self.io_queue.get()
            if io_work is None:                # Sentinel/Poison Pill check
                self.io_queue.task_done()
                print("Received sentinel value. Exiting IO worker.")
                break

            print(f"Processing telemetry data for device {io_work.device_id}: {io_work.metric}")

            # Check if io task is an instance of TelemetryData
            if not isinstance(io_work, TelemetryData):
                print(f"Invalid IO work: {io_work}. Expected TelemetryData instance.")
                queue.task_done()
                continue


            # Worker needs to process the telemetry data and insert it asynchronously into the database

            try:
                device_id = io_work.device_id
                metric = io_work.metric
                # Process the telemetry data here and insert asynchronously into the database
                print(f"Processing telemetry data for device {device_id}: {metric}")
                # Simulate database insertion (replace with actual DB logic)
                # For example, using sqlite3 to insert into a database asynchronously
                conn = await sqlite3.connect('telemetry.db')
                cursor = await conn.cursor()
                await cursor.execute('''CREATE TABLE IF NOT EXISTS telemetry (device_id TEXT, metric TEXT)''')
                await cursor.execute('INSERT INTO telemetry (device_id, metric) VALUES (?, ?)', (device_id, str(metric)))
                await conn.commit()
                await conn.close()
                print(f"Finished processing telemetry data for device {device_id}.") 
                
            finally:
                await queue.task_done()                  # Unblocks self.io_queue.join()  
        



async def main():

    # Create an instance of the IoWorkerQueue and start the IO worker
    io_worker_queue = IoWorkerQueue()

    # Simulate producing telemetry data
    for i in range(100):
        device_id = f"device_{i}"
        metric = {"temperature": 20 + i, "humidity": 50 + i}
        await io_worker_queue.produce_io_telemetry(device_id, metric)

    # Wait for all IO tasks to complete
    await io_worker_queue.io_queue.join()

    # Send a sentinel value to stop the IO workers
    for _ in range(io_worker_queue.num_workers):
        await io_worker_queue.io_queue.put(None)    
