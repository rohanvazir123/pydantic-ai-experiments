"""SQLModel models shared across the API, the store, and Temporal payloads.

`Order` is a SQLModel `table=True` model — one class is both the ORM table and the
Pydantic schema used at the API edge and in Temporal activity payloads. The other
models are non-table SQLModels (request/response DTOs).
"""

from __future__ import annotations

from sqlalchemy import Column
from sqlalchemy import Enum as SAEnum
from sqlmodel import Field, SQLModel

from app.domain import Decision, OrderStatus


class OrderInput(SQLModel):
    """Request body for creating an order."""

    item: str = Field(min_length=1)
    quantity: int = Field(gt=0)
    unit_price_cents: int = Field(ge=0)


class Order(SQLModel, table=True):
    """A persisted order (also the payload activities load/return).

    Status is stored as a portable VARCHAR (`native_enum=False`) so the same model
    works on SQLite (tests) and Postgres (prod) and round-trips back to OrderStatus.
    """

    id: str = Field(primary_key=True)
    item: str
    quantity: int
    unit_price_cents: int
    total_cents: int
    status: OrderStatus = Field(
        sa_column=Column(SAEnum(OrderStatus, native_enum=False), nullable=False)
    )
    created_at: str


class OrderResult(SQLModel):
    """Terminal result returned by the workflow."""

    order_id: str
    status: OrderStatus


class ApprovalInput(SQLModel):
    """Request body for the human-approval endpoint."""

    decision: Decision
