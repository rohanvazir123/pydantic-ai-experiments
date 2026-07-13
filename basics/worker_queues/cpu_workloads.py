# =====================================================================
# 1. IO-BOUND (asyncio: Telemetry APIs, Webhooks, DB Logs)
# =====================================================================
import asyncio
import multiprocessing

from pydantic import BaseModel

class ImageProcessingRequest(BaseModel):
    image_id: str
    image_data: bytes 


class CpuWorkerQueue:
    def __init__(self):

        # Singleton multiprocessing queue for CPU-bound tasks
        self.cpu_queue = multiprocessing.JoinableQueue()

        # Start CPU worker processes based on the number of CPU cores available
        self.num_cpu_workers = multiprocessing.cpu_count()
        self.cpu_workers = [multiprocessing.Process(target=self.process_cpu_tasks) for _ in range(self.num_cpu_workers)]

        print(f"Starting {self.num_cpu_workers} CPU worker processes.")
        for worker in self.cpu_workers:
            worker.start()


    # =====================================================================
    # 2. CPU-BOUND (multiprocessing: Parsing, Aggregating, Anomaly Check)
    # =====================================================================

    # The Consumer Pool Loop
    def process_cpu_tasks(self):
        while True:
            payload = self.cpu_queue.get()
            if payload is None:                # Sentinel/Poison Pill check
                self.cpu_queue.task_done()
                break

            # Check if payload is an instance of ImageProcessingRequest
            if not isinstance(payload, ImageProcessingRequest):
                print(f"Invalid CPU task: {payload}. Expected ImageProcessingRequest instance.")
                self.cpu_queue.task_done()
                continue

            # Worker needs to process the image data (CPU-bound task)
            try:
                print(f"Processing image data for image ID: {payload.image_id}")
                image_id = payload.image_id
                image_data = payload.image_data
                # Process the image data here (e.g., decoding, resizing, filtering)
                print(f"Finished processing image data for image ID: {image_id}")
                
            finally:
                print(f"Marking CPU task for image ID: {payload.image_id} as done.")
                self.cpu_queue.task_done()                  # Unblocks self.cpu_queue.join()

    # The Producer Entry Point
    def insert_cpu_tasks(self, raw_payloads: list[ImageProcessingRequest]):
        # Direct queue insertion. Blocks thread if queue maxsize is reached.
        for payload in raw_payloads:
            self.cpu_queue.put(payload)


# The main function to run the CPU worker processes only. 
# This is separate from the asyncio event loop for IO-bound tasks.
async def main():

    # Create a multiprocessing queue for CPU-bound tasks and start the CPU worker processes
    cpu_queue = CpuWorkerQueue()

    # Insert CPU-bound tasks into the queue
    raw_payloads = [ImageProcessingRequest(image_id=f"image_{i}", image_data=b"fake_image_data") for i in range(20)]
    cpu_queue.insert_cpu_tasks(raw_payloads) 

    # Wait for all CPU tasks to complete
    for cpu_worker in cpu_queue.cpu_workers:
        cpu_worker.join()


    # Send a sentinel value to stop the CPU worker processes
    # Need just one sentinel per worker process to signal them to exit
    for _ in range(cpu_queue.num_cpu_workers):
        cpu_queue.cpu_queue.put(None)
    
    

