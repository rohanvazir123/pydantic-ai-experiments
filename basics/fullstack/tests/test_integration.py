"""Integration tests for the real TemporalWorkflowStarter (client.py) wiring."""

from __future__ import annotations

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from app.domain import Decision, OrderStatus
from app.models import Order, OrderResult
from app.store.memory import InMemoryOrderRepository
from app.temporal.activities import OrderActivities
from app.temporal.client import TemporalWorkflowStarter
from app.temporal.workflow import OrderWorkflow

TASK_QUEUE = "it-order-queue"


def _order(order_id: str, *, unit_price_cents: int) -> Order:
    return Order(
        id=order_id, item="Widget", quantity=1, unit_price_cents=unit_price_cents,
        total_cents=unit_price_cents, status=OrderStatus.PENDING,
        created_at="2026-07-05T00:00:00+00:00",
    )


def _worker(env: WorkflowEnvironment, repo: InMemoryOrderRepository) -> Worker:
    acts = OrderActivities(repo)
    return Worker(
        env.client, task_queue=TASK_QUEUE, workflows=[OrderWorkflow],
        activities=[acts.load_order, acts.mark_status],
    )


async def test_starter_confirms_low_value(
    temporal_env: WorkflowEnvironment, repo: InMemoryOrderRepository
) -> None:
    await repo.create(_order("i1", unit_price_cents=500))
    starter = TemporalWorkflowStarter(temporal_env.client, TASK_QUEUE)
    async with _worker(temporal_env, repo):
        await starter.start_order("i1")
        handle = temporal_env.client.get_workflow_handle("order-i1", result_type=OrderResult)
        result = await handle.result()
    assert result.status is OrderStatus.CONFIRMED


async def test_starter_high_value_approve(
    temporal_env: WorkflowEnvironment, repo: InMemoryOrderRepository
) -> None:
    await repo.create(_order("i2", unit_price_cents=200_000))
    starter = TemporalWorkflowStarter(temporal_env.client, TASK_QUEUE)
    async with _worker(temporal_env, repo):
        await starter.start_order("i2")
        await starter.approve("i2", Decision.APPROVED)  # signal buffered until the wait
        handle = temporal_env.client.get_workflow_handle("order-i2", result_type=OrderResult)
        result = await handle.result()
    assert result.status is OrderStatus.CONFIRMED
