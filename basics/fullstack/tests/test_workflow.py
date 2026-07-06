"""Workflow tests against Temporal's in-process time-skipping test server.

Covers: low-value auto-confirm, invalid-order reject, and all three HIL outcomes
(approve, reject, SLA-timeout). Time-skipping fast-forwards the 2-day SLA timer so
the timeout test runs in milliseconds.
"""

from __future__ import annotations

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from app.domain import Decision, OrderStatus
from app.models import Order
from app.store.memory import InMemoryOrderRepository
from app.temporal.activities import OrderActivities
from app.temporal.workflow import OrderWorkflow

TASK_QUEUE = "test-order-queue"


def _order(order_id: str, *, quantity: int = 1, unit_price_cents: int = 500) -> Order:
    return Order(
        id=order_id, item="Widget", quantity=quantity, unit_price_cents=unit_price_cents,
        total_cents=quantity * unit_price_cents, status=OrderStatus.PENDING,
        created_at="2026-07-05T00:00:00+00:00",
    )


def _worker(env: WorkflowEnvironment, repo: InMemoryOrderRepository) -> Worker:
    acts = OrderActivities(repo)
    return Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[OrderWorkflow],
        activities=[acts.load_order, acts.mark_status],
    )


async def test_low_value_confirmed(
    temporal_env: WorkflowEnvironment, repo: InMemoryOrderRepository
) -> None:
    await repo.create(_order("low", quantity=2, unit_price_cents=500))  # $10 total
    async with _worker(temporal_env, repo):
        result = await temporal_env.client.execute_workflow(
            OrderWorkflow.run, "low", id="wf-low", task_queue=TASK_QUEUE
        )
    assert result.status is OrderStatus.CONFIRMED
    stored = await repo.get("low")
    assert stored is not None and stored.status is OrderStatus.CONFIRMED


async def test_invalid_order_rejected(
    temporal_env: WorkflowEnvironment, repo: InMemoryOrderRepository
) -> None:
    bad = _order("bad").model_copy(update={"quantity": 0})  # fails domain validation
    await repo.create(bad)
    async with _worker(temporal_env, repo):
        result = await temporal_env.client.execute_workflow(
            OrderWorkflow.run, "bad", id="wf-bad", task_queue=TASK_QUEUE
        )
    assert result.status is OrderStatus.REJECTED


async def test_high_value_approved(
    temporal_env: WorkflowEnvironment, repo: InMemoryOrderRepository
) -> None:
    await repo.create(_order("hi", quantity=1, unit_price_cents=200_000))  # $2,000
    async with _worker(temporal_env, repo):
        handle = await temporal_env.client.start_workflow(
            OrderWorkflow.run, "hi", id="wf-hi", task_queue=TASK_QUEUE
        )
        await handle.signal(OrderWorkflow.submit_approval, Decision.APPROVED)
        result = await handle.result()
    assert result.status is OrderStatus.CONFIRMED


async def test_high_value_rejected_by_signal(
    temporal_env: WorkflowEnvironment, repo: InMemoryOrderRepository
) -> None:
    await repo.create(_order("hr", quantity=1, unit_price_cents=200_000))
    async with _worker(temporal_env, repo):
        handle = await temporal_env.client.start_workflow(
            OrderWorkflow.run, "hr", id="wf-hr", task_queue=TASK_QUEUE
        )
        await handle.signal(OrderWorkflow.submit_approval, Decision.REJECTED)
        result = await handle.result()
    assert result.status is OrderStatus.REJECTED


async def test_high_value_sla_timeout_rejected(
    temporal_env: WorkflowEnvironment, repo: InMemoryOrderRepository
) -> None:
    await repo.create(_order("to", quantity=1, unit_price_cents=200_000))
    async with _worker(temporal_env, repo):
        # No signal sent — time-skipping fires the 2-day SLA timer automatically.
        result = await temporal_env.client.execute_workflow(
            OrderWorkflow.run, "to", id="wf-to", task_queue=TASK_QUEUE
        )
    assert result.status is OrderStatus.REJECTED
    stored = await repo.get("to")
    assert stored is not None and stored.status is OrderStatus.REJECTED
