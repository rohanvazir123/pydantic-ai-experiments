import asyncio
from temporalio.client import Client
from temporalio.worker import Worker
from workflows import OrderProcessingWorkflow
from activities import charge_payment, send_receipt

async def main():
    # Connect to local Temporal server (Standard port: 7233)
    client = await Client.connect("localhost:7233")

    # Run the worker listening on a specific task queue
    worker = Worker(
        client,
        task_queue="order-task-queue",
        workflows=[OrderProcessingWorkflow],
        activities=[charge_payment, send_receipt],
    )

    print("Worker is running. Press Ctrl+C to exit.")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
