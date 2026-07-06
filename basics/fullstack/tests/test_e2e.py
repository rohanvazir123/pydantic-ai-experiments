"""Full-stack end-to-end: real HTTP -> FastAPI -> Temporal -> worker -> repo -> HTTP.

Uses httpx's in-process ASGI transport (fully async, no thread) against the real
app wired to a real WorkflowStarter and a running worker on the test server.
"""

from __future__ import annotations

import httpx
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from app.api.app import create_app
from app.domain import OrderStatus
from app.store.memory import InMemoryOrderRepository
from app.temporal.activities import OrderActivities
from app.temporal.client import TemporalWorkflowStarter
from app.temporal.workflow import OrderWorkflow

TASK_QUEUE = "e2e-order-queue"


def _worker(env: WorkflowEnvironment, repo: InMemoryOrderRepository) -> Worker:
    acts = OrderActivities(repo)
    return Worker(
        env.client, task_queue=TASK_QUEUE, workflows=[OrderWorkflow],
        activities=[acts.load_order, acts.mark_status],
    )


async def test_http_full_stack_low_value_confirms(
    temporal_env: WorkflowEnvironment, repo: InMemoryOrderRepository
) -> None:
    starter = TemporalWorkflowStarter(temporal_env.client, TASK_QUEUE)
    app = create_app(repo, starter)
    transport = httpx.ASGITransport(app=app)
    async with _worker(temporal_env, repo):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/orders", json={"item": "Widget", "quantity": 2, "unit_price_cents": 500}
            )
            assert resp.status_code == 202
            order_id = resp.json()["order_id"]

            # Let the workflow finish (no timer on the low-value path).
            await temporal_env.client.get_workflow_handle(f"order-{order_id}").result()

            got = await ac.get(f"/api/orders/{order_id}")
            assert got.status_code == 200
            assert got.json()["status"] == OrderStatus.CONFIRMED.value
