"""
Tests for OrderFulfillmentSagaWorkflow.

Six scenarios:
  1. Happy path              — all 4 stages succeed, order fulfilled
  2. Payment fails           — charge_payment raises → no saga chain, immediate cancel
  3. Inventory fails         — reserve_inventory raises → refund only, no penalty
                                (confirmation email never sent, so no promise was broken)
  4. Confirmation fails      — send_confirmation_email raises → release + refund, no
                                penalty (the email that would create the promise never sent)
  5. Shipping fails          — ship_order raises after confirmation went out → release +
                                refund WITH the 10% broken-delivery-promise penalty
  6. Rollback order check    — verifies exact reverse order via spy activities

One ephemeral Temporal server is shared across the module via the temporal_env fixture.
"""
from __future__ import annotations

import pytest

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio import activity as _activity

from ..activities import OrderActivities
from ..models import OrderInput, OrderReport, RefundInput, StepResult
from ..workflows import OrderFulfillmentSagaWorkflow

pytestmark = pytest.mark.asyncio(loop_scope="module")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TASK_QUEUE = "test-order-saga-queue"

ORDER = OrderInput(order_id="order-001", amount=200.0, target_ship_date="2026-07-10")


def _all_activities(order_acts: OrderActivities) -> list[object]:
    return [
        order_acts.charge_payment,
        order_acts.refund_payment,
        order_acts.reserve_inventory,
        order_acts.release_inventory,
        order_acts.send_confirmation_email,
        order_acts.ship_order,
    ]


# ---------------------------------------------------------------------------
# Test 1: Happy path — all stages succeed
# ---------------------------------------------------------------------------


async def test_happy_path_order_fulfilled(temporal_env: WorkflowEnvironment) -> None:
    """All 4 stages succeed. Report: succeeded=True, no compensations."""
    order_acts = OrderActivities()

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[OrderFulfillmentSagaWorkflow],
        activities=_all_activities(order_acts),
    ):
        result_json = await temporal_env.client.execute_workflow(
            OrderFulfillmentSagaWorkflow.run,
            ORDER.model_dump_json(),
            id="test-order-saga-001",
            task_queue=TASK_QUEUE,
        )

    report = OrderReport.model_validate_json(result_json)
    assert report.succeeded is True
    assert report.final_status == "fulfilled"
    assert report.compensations_run == []
    assert report.aborted_at is None
    assert len(report.completed_stages) == 4


# ---------------------------------------------------------------------------
# Test 2: Payment fails → no saga chain, immediate cancel
# ---------------------------------------------------------------------------


async def test_charge_payment_fails_no_compensations(temporal_env: WorkflowEnvironment) -> None:
    """charge_payment fails before anything completes → no rollback needed."""
    order_acts = OrderActivities(scenario={"charge_payment": False})

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[OrderFulfillmentSagaWorkflow],
        activities=_all_activities(order_acts),
    ):
        result_json = await temporal_env.client.execute_workflow(
            OrderFulfillmentSagaWorkflow.run,
            ORDER.model_dump_json(),
            id="test-order-saga-002",
            task_queue=TASK_QUEUE,
        )

    report = OrderReport.model_validate_json(result_json)
    assert report.succeeded is False
    assert report.aborted_at == "charge_payment"
    assert report.compensations_run == []
    assert report.completed_stages == []
    assert report.refund_amount is None


# ---------------------------------------------------------------------------
# Test 3: Inventory fails → refund only, no penalty
# ---------------------------------------------------------------------------


async def test_inventory_fails_refunds_without_penalty(temporal_env: WorkflowEnvironment) -> None:
    """reserve_inventory raises → saga chain = [charge_payment] → refund_payment,
    no penalty since the confirmation email (the delivery promise) never went out."""
    order_acts = OrderActivities(scenario={"reserve_inventory": False})

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[OrderFulfillmentSagaWorkflow],
        activities=_all_activities(order_acts),
    ):
        result_json = await temporal_env.client.execute_workflow(
            OrderFulfillmentSagaWorkflow.run,
            ORDER.model_dump_json(),
            id="test-order-saga-003",
            task_queue=TASK_QUEUE,
        )

    report = OrderReport.model_validate_json(result_json)
    assert report.succeeded is False
    assert report.aborted_at == "reserve_inventory"
    assert report.compensations_run == ["refund_payment"]
    assert report.refund_amount == ORDER.amount  # no penalty applied


# ---------------------------------------------------------------------------
# Test 4: Confirmation email fails → release + refund, no penalty
# ---------------------------------------------------------------------------


async def test_confirmation_email_fails_refunds_without_penalty(
    temporal_env: WorkflowEnvironment,
) -> None:
    """send_confirmation_email raises → saga chain = [charge_payment, reserve_inventory]
    → release_inventory then refund_payment, no penalty (the email itself never sent)."""
    order_acts = OrderActivities(scenario={"send_confirmation_email": False})

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[OrderFulfillmentSagaWorkflow],
        activities=_all_activities(order_acts),
    ):
        result_json = await temporal_env.client.execute_workflow(
            OrderFulfillmentSagaWorkflow.run,
            ORDER.model_dump_json(),
            id="test-order-saga-004",
            task_queue=TASK_QUEUE,
        )

    report = OrderReport.model_validate_json(result_json)
    assert report.succeeded is False
    assert report.aborted_at == "send_confirmation_email"
    assert report.compensations_run == ["release_inventory", "refund_payment"]
    assert report.refund_amount == ORDER.amount  # no penalty applied


# ---------------------------------------------------------------------------
# Test 5: Shipping fails → refund WITH broken-delivery-promise penalty
# ---------------------------------------------------------------------------


async def test_shipping_fails_refunds_with_penalty(temporal_env: WorkflowEnvironment) -> None:
    """ship_order raises after the confirmation email already promised a ship
    date → saga chain = [charge_payment, reserve_inventory] → release_inventory
    then refund_payment WITH the 10% penalty. This is the motivating scenario:
    payment succeeded, a promise was made, a later step fails, and the
    compensation for the earlier successful step reflects that broken promise."""
    order_acts = OrderActivities(scenario={"ship_order": False})

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[OrderFulfillmentSagaWorkflow],
        activities=_all_activities(order_acts),
    ):
        result_json = await temporal_env.client.execute_workflow(
            OrderFulfillmentSagaWorkflow.run,
            ORDER.model_dump_json(),
            id="test-order-saga-005",
            task_queue=TASK_QUEUE,
        )

    report = OrderReport.model_validate_json(result_json)
    assert report.succeeded is False
    assert report.aborted_at == "ship_order"
    assert report.compensations_run == ["release_inventory", "refund_payment"]
    assert report.refund_amount == round(ORDER.amount * 1.10, 2)  # 200 -> 220.0


# ---------------------------------------------------------------------------
# Test 6: Rollback order check — verify exact reverse order via spies
# ---------------------------------------------------------------------------


async def test_rollback_order_is_exactly_reversed(temporal_env: WorkflowEnvironment) -> None:
    """When ship_order fails, compensations must run in exact reverse insertion
    order: release_inventory (reserve_inventory's compensation) before
    refund_payment (charge_payment's compensation)."""
    call_order: list[str] = []

    @_activity.defn(name="release_inventory")
    async def spy_release_inventory(order_id: str) -> str:
        call_order.append("release_inventory")
        return StepResult(
            stage="release_inventory", success=True, message="done"
        ).model_dump_json()

    @_activity.defn(name="refund_payment")
    async def spy_refund_payment(refund: RefundInput) -> str:
        call_order.append("refund_payment")
        return StepResult(
            stage="refund_payment", success=True, message="done", refund_amount=220.0
        ).model_dump_json()

    order_acts = OrderActivities(scenario={"ship_order": False})

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[OrderFulfillmentSagaWorkflow],
        activities=[
            order_acts.charge_payment,
            spy_refund_payment,
            order_acts.reserve_inventory,
            spy_release_inventory,
            order_acts.send_confirmation_email,
            order_acts.ship_order,
        ],
    ):
        result_json = await temporal_env.client.execute_workflow(
            OrderFulfillmentSagaWorkflow.run,
            ORDER.model_dump_json(),
            id="test-order-saga-006",
            task_queue=TASK_QUEUE,
        )

    report = OrderReport.model_validate_json(result_json)
    assert report.succeeded is False
    assert call_order == ["release_inventory", "refund_payment"]
    assert report.compensations_run == call_order
