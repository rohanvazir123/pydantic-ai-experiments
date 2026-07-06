"""Temporal activities — the ONLY place workflow-driven I/O happens.

Activities are methods on a class holding the repository, so dependencies are
injected (real Postgres in prod, in-memory in tests) and each method is a plain
async function that unit-tests directly.
"""

from __future__ import annotations

from temporalio import activity
from temporalio.exceptions import ApplicationError

from app.domain import OrderStatus
from app.models import Order
from app.store.base import OrderRepository


class OrderActivities:
    def __init__(self, repo: OrderRepository) -> None:
        self._repo = repo

    @activity.defn
    async def load_order(self, order_id: str) -> Order:
        order = await self._repo.get(order_id)
        if order is None:
            # Missing data is not a transient error — don't retry.
            raise ApplicationError(f"order not found: {order_id}", non_retryable=True)
        return order

    @activity.defn
    async def mark_status(self, order_id: str, status: OrderStatus) -> None:
        await self._repo.set_status(order_id, status)
