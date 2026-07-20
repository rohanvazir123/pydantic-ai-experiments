import asyncio
import uuid
from temporalio.client import Client
from workflows import OrderProcessingWorkflow, OrderDetails

async def main():
    client = await Client.connect("localhost:7233")
    
    order = OrderDetails(
        order_id=f"order-{uuid.uuid4().hex[:6]}",
        amount=150.0,
        user_email="user@example.com"
    )

    # Start the Workflow asynchronously
    handle = await client.start_workflow(
        OrderProcessingWorkflow.run,
        order,
        id=f"workflow-{order.order_id}",
        task_queue="order-task-queue",
    )

    print(f"Workflow started with ID: {handle.id}")
    
    # Example Query
    status = await handle.query(OrderProcessingWorkflow.get_status)
    print(f"Current State: {status}")

    # Example Signal (Usually called by an external service/UI, not the same script)
    input("Press Enter once the order has been manually approved...")
    await handle.signal(OrderProcessingWorkflow.approve_order)

    # Await Final Result
    result = await handle.result()
    print(f"Workflow Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
