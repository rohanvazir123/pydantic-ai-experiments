"""
Temporal activities for the Order Fulfillment Saga workflow.

Simulated order-fulfillment operations driven by a scenario dict. No LLM
involved — this example is a pure saga-pattern demonstration: a downstream
step (ship_order) fails and earlier successful steps get compensated.
"""
from __future__ import annotations

from temporalio import activity

from .models import OrderInput, RefundInput, StepResult

_DEFAULT_SCENARIO: dict[str, bool] = {
    "charge_payment": True,
    "reserve_inventory": True,
    "send_confirmation_email": True,
    "ship_order": True,
}

# Refund penalty applied when the customer already received a confirmation
# email promising a ship date, and that promise was then broken.
BROKEN_PROMISE_PENALTY_PCT = 0.10


class OrderActivities:
    """Simulated order-fulfillment operations. Pass a partial scenario dict to override defaults."""

    def __init__(self, scenario: dict[str, bool] | None = None) -> None:
        self._scenario: dict[str, bool] = {**_DEFAULT_SCENARIO, **(scenario or {})}

    def _check(self, stage: str, order_id: str, detail: str) -> None:
        if not self._scenario.get(stage, True):
            raise RuntimeError(f"{stage} failed for {order_id} ({detail})")

    @activity.defn(name="charge_payment")
    async def charge_payment(self, order: OrderInput) -> str:
        self._check("charge_payment", order.order_id, "payment declined, simulated")
        return StepResult(
            stage="charge_payment",
            success=True,
            message=f"charged ${order.amount:.2f} for {order.order_id}",
        ).model_dump_json()

    @activity.defn(name="refund_payment")
    async def refund_payment(self, refund: RefundInput) -> str:
        total = round(refund.amount * (1 + refund.penalty_pct), 2)
        note = (
            f" (includes {refund.penalty_pct:.0%} broken-delivery-promise penalty)"
            if refund.penalty_pct
            else ""
        )
        return StepResult(
            stage="refund_payment",
            success=True,
            message=f"refunded ${total:.2f} for {refund.order_id}{note}",
            refund_amount=total,
        ).model_dump_json()

    @activity.defn(name="reserve_inventory")
    async def reserve_inventory(self, order: OrderInput) -> str:
        self._check("reserve_inventory", order.order_id, "out of stock, simulated")
        return StepResult(
            stage="reserve_inventory",
            success=True,
            message=f"inventory reserved for {order.order_id}",
        ).model_dump_json()

    @activity.defn(name="release_inventory")
    async def release_inventory(self, order_id: str) -> str:
        return StepResult(
            stage="release_inventory",
            success=True,
            message=f"inventory released for {order_id}",
        ).model_dump_json()

    @activity.defn(name="send_confirmation_email")
    async def send_confirmation_email(self, order: OrderInput) -> str:
        self._check(
            "send_confirmation_email", order.order_id, "email service down, simulated"
        )
        return StepResult(
            stage="send_confirmation_email",
            success=True,
            message=(
                f"confirmation sent for {order.order_id}, "
                f"target ship date {order.target_ship_date}"
            ),
        ).model_dump_json()

    @activity.defn(name="ship_order")
    async def ship_order(self, order: OrderInput) -> str:
        self._check(
            "ship_order",
            order.order_id,
            f"warehouse could not fulfill by {order.target_ship_date}, simulated",
        )
        return StepResult(
            stage="ship_order", success=True, message=f"shipped {order.order_id}"
        ).model_dump_json()
