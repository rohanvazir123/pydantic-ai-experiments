"""Temporal client helpers + the WorkflowStarter the API depends on."""

from __future__ import annotations

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from app.domain import Decision
from app.temporal.workflow import OrderWorkflow


async def connect(target: str) -> Client:
    """Connect a client that serializes Pydantic models in payloads."""
    return await Client.connect(target, data_converter=pydantic_data_converter)


class TemporalWorkflowStarter:
    """Production WorkflowStarter: starts workflows and delivers approval signals.

    workflow_id = f"order-{order_id}" gives natural dedup — starting the same order
    twice is rejected by Temporal.
    """

    def __init__(self, client: Client, task_queue: str) -> None:
        self._client = client
        self._task_queue = task_queue

    async def start_order(self, order_id: str) -> None:
        await self._client.start_workflow(
            OrderWorkflow.run, order_id,
            id=f"order-{order_id}", task_queue=self._task_queue,
        )

    async def approve(self, order_id: str, decision: Decision) -> None:
        handle = self._client.get_workflow_handle(f"order-{order_id}")
        await handle.signal(OrderWorkflow.submit_approval, decision)
