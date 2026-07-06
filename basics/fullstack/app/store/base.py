"""The repository interface. Both the API and the Temporal activities depend on
this Protocol, never on a concrete store — so tests inject the in-memory impl and
production injects Postgres with zero code changes."""

from __future__ import annotations

from typing import Protocol

from app.domain import OrderStatus
from app.models import Order


class OrderRepository(Protocol):
    async def create(self, order: Order) -> None: ...

    async def get(self, order_id: str) -> Order | None: ...

    async def set_status(self, order_id: str, status: OrderStatus) -> None: ...

    async def list(self) -> list[Order]: ...
