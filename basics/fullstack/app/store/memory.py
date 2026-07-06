"""In-memory OrderRepository — used by the whole test suite (no DB required)."""

from __future__ import annotations

import asyncio

from app.domain import OrderStatus
from app.models import Order


class InMemoryOrderRepository:
    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self._lock = asyncio.Lock()

    async def create(self, order: Order) -> None:
        async with self._lock:
            self._orders[order.id] = order

    async def get(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    async def set_status(self, order_id: str, status: OrderStatus) -> None:
        async with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                raise KeyError(order_id)
            self._orders[order_id] = order.model_copy(update={"status": status})

    async def list(self) -> list[Order]:
        return list(self._orders.values())
