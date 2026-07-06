"""Postgres OrderRepository (asyncpg, SQL-first) — the production store.

Not exercised by the unit suite (which uses the in-memory repo); wired in by the
worker and API entrypoints against a real database.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from app.domain import OrderStatus
from app.models import Order

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS orders (
    id               TEXT PRIMARY KEY,
    item             TEXT NOT NULL,
    quantity         INTEGER NOT NULL,
    unit_price_cents BIGINT NOT NULL,
    total_cents      BIGINT NOT NULL,
    status           TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
"""


class PostgresOrderRepository:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn)
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("repository not connected; call connect() first")
        return self._pool

    async def create(self, order: Order) -> None:
        await self.pool.execute(
            """INSERT INTO orders (id, item, quantity, unit_price_cents, total_cents,
                                   status, created_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7)""",
            order.id, order.item, order.quantity, order.unit_price_cents,
            order.total_cents, order.status.value, order.created_at,
        )

    async def get(self, order_id: str) -> Order | None:
        row: Any = await self.pool.fetchrow("SELECT * FROM orders WHERE id = $1", order_id)
        return _row_to_order(row) if row is not None else None

    async def set_status(self, order_id: str, status: OrderStatus) -> None:
        result = await self.pool.execute(
            "UPDATE orders SET status = $2 WHERE id = $1", order_id, status.value
        )
        if result.endswith("0"):
            raise KeyError(order_id)

    async def list(self) -> list[Order]:
        rows = await self.pool.fetch("SELECT * FROM orders ORDER BY created_at DESC")
        return [_row_to_order(r) for r in rows]


def _row_to_order(row: Any) -> Order:
    return Order(
        id=row["id"],
        item=row["item"],
        quantity=row["quantity"],
        unit_price_cents=row["unit_price_cents"],
        total_cents=row["total_cents"],
        status=OrderStatus(row["status"]),
        created_at=row["created_at"],
    )
