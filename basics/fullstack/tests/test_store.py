"""Unit tests for the in-memory repository."""

from __future__ import annotations

import pytest

from app.domain import OrderStatus
from app.models import Order
from app.store.memory import InMemoryOrderRepository


def _order(order_id: str = "o1", status: OrderStatus = OrderStatus.PENDING) -> Order:
    return Order(
        id=order_id, item="Widget", quantity=2, unit_price_cents=500,
        total_cents=1000, status=status, created_at="2026-07-05T00:00:00+00:00",
    )


async def test_create_and_get(repo: InMemoryOrderRepository) -> None:
    await repo.create(_order())
    got = await repo.get("o1")
    assert got is not None
    assert got.item == "Widget"


async def test_get_missing_returns_none(repo: InMemoryOrderRepository) -> None:
    assert await repo.get("nope") is None


async def test_set_status(repo: InMemoryOrderRepository) -> None:
    await repo.create(_order())
    await repo.set_status("o1", OrderStatus.CONFIRMED)
    got = await repo.get("o1")
    assert got is not None
    assert got.status is OrderStatus.CONFIRMED


async def test_set_status_missing_raises(repo: InMemoryOrderRepository) -> None:
    with pytest.raises(KeyError):
        await repo.set_status("nope", OrderStatus.CONFIRMED)


async def test_list(repo: InMemoryOrderRepository) -> None:
    await repo.create(_order("a"))
    await repo.create(_order("b"))
    ids = {o.id for o in await repo.list()}
    assert ids == {"a", "b"}
