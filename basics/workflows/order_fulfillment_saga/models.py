from __future__ import annotations

from pydantic import BaseModel


class OrderInput(BaseModel):
    order_id: str
    amount: float
    target_ship_date: str


class RefundInput(BaseModel):
    order_id: str
    amount: float
    penalty_pct: float = 0.0


class StepResult(BaseModel):
    stage: str
    success: bool
    message: str
    refund_amount: float | None = None


class OrderReport(BaseModel):
    order_id: str
    completed_stages: list[str]
    compensations_run: list[str]
    succeeded: bool
    aborted_at: str | None = None
    final_status: str
    refund_amount: float | None = None
