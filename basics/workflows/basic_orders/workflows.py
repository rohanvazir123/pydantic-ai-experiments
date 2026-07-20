import asyncio
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict
from temporalio import workflow, exceptions
from temporalio.common import RetryPolicy

# Import activities
with workflow.unsafe.imports_passed_through():
    from activities import charge_payment, send_receipt

@dataclass
class OrderDetails:
    order_id: str
    amount: float
    user_email: str

@workflow.defn
class OrderProcessingWorkflow:
    def __init__(self) -> None:
        self._status = "Initialized"
        self._is_approved = False
        self._payment_result = None

    # 1. QUERIES: Inspect the workflow state in real-time without modifying it
    @workflow.query
    def get_status(self) -> str:
        return self._status

    # 2. SIGNALS: Trigger asynchronous actions like human approval or external events
    @workflow.signal
    async def approve_order(self) -> None:
        self._is_approved = True

    # 3. UPDATES: Accept input and return a response (or execute logic on demand)
    @workflow.update
    async def update_delivery_address(self, new_address: str) -> str:
        if self._status == "Completed":
            raise ValueError("Cannot change address after order is shipped")
        return f"Address updated to {new_address}"

    # 4. WORKFLOW RUN: Orchestrates the core pipeline
    @workflow.run
    async def run(self, order: OrderDetails) -> Dict[str, Any]:

        # 
        compensations = []

        self._status = "Waiting for Approval"

        # 5. DURABLE TIMERS: Pauses without consuming compute resources
        # Times out and proceeds to cancel if no approval comes within 24 hours
        try:
            await workflow.wait_condition(lambda: self._is_approved, timeout=timedelta(hours=24))
        except exceptions.TimeoutError:
            self._status = "Timed out waiting for approval"
            return {"status": self._status}

        self._status = "Processing Payment"

        try:
        
            # 6. ACTIVITIES: Invoke a business action with strict retry policies
            self._payment_result = await workflow.execute_activity(
                charge_payment,
                order.amount,
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            self._status = "Sending Receipt"
            receipt_result = await workflow.execute_activity(
                send_receipt,
                {"email": order.user_email, "amount": order.amount},
                start_to_close_timeout=timedelta(seconds=10),
            )

            self._status = "Completed"

            compensations.append((refund_payment, order.amount))

            # Step 2: Create Shipment (Assume this fails)
            await workflow.execute_activity(lambda shipment: shipment, ...) 

            return {
                "status": self._status,
                "payment": self._payment_result,
                "receipt": receipt_result
            }
        
        except Exception as e:
            self._status = "Failed. Rolling back..."
            # Run compensations in reverse order
            for activity, args in reversed(compensations):
                # Use a relaxed timeout or customized retry policy for compensations
                await workflow.execute_activity(
                    activity, 
                    args, 
                    start_to_close_timeout=timedelta(seconds=30)
                )
            raise e
