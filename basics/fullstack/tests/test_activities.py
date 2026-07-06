"""Unit tests for Temporal activities — called directly with an in-memory repo."""

from __future__ import annotations

import pytest
from temporalio.exceptions import ApplicationError

from app.domain import OrderStatus
from app.models import Order
from app.store.memory import InMemoryOrderRepository
from app.temporal.activities import OrderActivities


def _order(order_id: str = "o1") -> Order:
    return Order(
        id=order_id, item="Widget", quantity=2, unit_price_cents=500,
        total_cents=1000, status=OrderStatus.PENDING, created_at="2026-07-05T00:00:00+00:00",
    )


async def test_load_order(repo: InMemoryOrderRepository) -> None:
    await repo.create(_order())
    acts = OrderActivities(repo)
    order = await acts.load_order("o1")
    assert order.id == "o1"


async def test_load_order_missing_raises(repo: InMemoryOrderRepository) -> None:
    acts = OrderActivities(repo)
    with pytest.raises(ApplicationError):
        await acts.load_order("nope")


async def test_mark_status(repo: InMemoryOrderRepository) -> None:
    await repo.create(_order())
    acts = OrderActivities(repo)
    await acts.mark_status("o1", OrderStatus.CONFIRMED)
    got = await repo.get("o1")
    assert got is not None
    assert got.status is OrderStatus.CONFIRMED
