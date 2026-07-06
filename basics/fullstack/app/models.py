"""Pydantic DTOs shared across the API, the store, and Temporal payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain import Decision, OrderStatus


class OrderInput(BaseModel):
    """Request body for creating an order."""

    item: str = Field(min_length=1)
    quantity: int = Field(gt=0)
    unit_price_cents: int = Field(ge=0)


class Order(BaseModel):
    """A persisted order (also the payload activities load/return)."""

    id: str
    item: str
    quantity: int
    unit_price_cents: int
    total_cents: int
    status: OrderStatus
    created_at: str


class OrderResult(BaseModel):
    """Terminal result returned by the workflow."""

    order_id: str
    status: OrderStatus


class ApprovalInput(BaseModel):
    """Request body for the human-approval endpoint."""

    decision: Decision
