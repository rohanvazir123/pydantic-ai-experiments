"""Tests for the production SQLModel repository, run against in-memory SQLite.

Exercises the real async SQLAlchemy/SQLModel code path (same code that runs on
Postgres in production) without needing a Postgres server.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from app.domain import OrderStatus
from app.models import Order
from app.store.sqlmodel_repo import SqlModelOrderRepository, create_engine, init_db


def _order(order_id: str = "o1", status: OrderStatus = OrderStatus.PENDING) -> Order:
    return Order(
        id=order_id, item="Widget", quantity=2, unit_price_cents=500,
        total_cents=1000, status=status, created_at="2026-07-06T00:00:00+00:00",
    )


@pytest_asyncio.fixture
async def sql_repo() -> AsyncIterator[SqlModelOrderRepository]:
    engine = create_engine("sqlite+aiosqlite://")  # in-memory, shared connection
    await init_db(engine)
    try:
        yield SqlModelOrderRepository(engine)
    finally:
        await engine.dispose()


async def test_create_and_get(sql_repo: SqlModelOrderRepository) -> None:
    await sql_repo.create(_order())
    got = await sql_repo.get("o1")
    assert got is not None
    assert got.item == "Widget"
    assert got.status is OrderStatus.PENDING  # enum round-trips


async def test_get_missing_returns_none(sql_repo: SqlModelOrderRepository) -> None:
    assert await sql_repo.get("nope") is None


async def test_set_status(sql_repo: SqlModelOrderRepository) -> None:
    await sql_repo.create(_order())
    await sql_repo.set_status("o1", OrderStatus.CONFIRMED)
    got = await sql_repo.get("o1")
    assert got is not None
    assert got.status is OrderStatus.CONFIRMED


async def test_set_status_missing_raises(sql_repo: SqlModelOrderRepository) -> None:
    with pytest.raises(KeyError):
        await sql_repo.set_status("nope", OrderStatus.CONFIRMED)


async def test_list(sql_repo: SqlModelOrderRepository) -> None:
    await sql_repo.create(_order("a"))
    await sql_repo.create(_order("b"))
    ids = {o.id for o in await sql_repo.list()}
    assert ids == {"a", "b"}
