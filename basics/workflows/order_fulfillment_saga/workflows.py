"""
OrderFulfillmentSagaWorkflow — sequential order pipeline with policy-aware cascade rollback.

Saga guarantees:
  - Each completed state-changing step registers a compensation.
  - On any failure, all registered compensations run in exact reverse order.
  - Unlike a plain mechanical undo, refund_payment is policy-aware: if the
    customer had already received a confirmation email promising a ship
    date, the refund includes a delivery-promise penalty
    (BROKEN_PROMISE_PENALTY_PCT) — the compensation reflects *why* the saga
    unwound, not just which action it's undoing.

Pipeline stages:
  1. charge_payment            → compensation: refund_payment
  2. reserve_inventory         → compensation: release_inventory
  3. send_confirmation_email   → stateless (no compensation) — but completing
                                  it means a delivery promise is now outstanding
  4. ship_order                → stateless; if the primary warehouse can't
                                  fulfill by the target date, this step fails
  5. ship_order_backup_warehouse → SELF-HEAL: tried once, forward-recovery
                                  attempt before giving up. If it also fails,
                                  the saga unwinds everything still on the chain

Self-heal vs. saga compensation — two different responses to failure:
  - Self-heal (step 5) is forward recovery: try an alternate path once
    before giving up.
  - Saga compensation (_rollback) is backward recovery: undo whatever
    already succeeded, because nothing else is left to try.

Temporal guarantees each step runs durably. Saga chain guarantees rollback order.
"""
from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

from .activities import BROKEN_PROMISE_PENALTY_PCT
from .models import OrderInput, OrderReport, RefundInput, StepResult

_TIMEOUT = timedelta(seconds=30)
_STEP_RETRY = RetryPolicy(maximum_attempts=2, initial_interval=timedelta(seconds=1))
_COMP_RETRY = RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=1))

# None = stateless / no rollback needed for that stage itself.
COMPENSATIONS: dict[str, str | None] = {
    "charge_payment": "refund_payment",
    "reserve_inventory": "release_inventory",
    "send_confirmation_email": None,
    "ship_order": None,
    "ship_order_backup_warehouse": None,
}


@workflow.defn(sandboxed=False)
class OrderFulfillmentSagaWorkflow:
    @workflow.run
    async def run(self, order_json: str) -> str:
        order = OrderInput.model_validate_json(order_json)
        completed: list[str] = []
        # SAGA: chain of (stage, compensation_activity_name) for completed,
        # reversible stages — unwound in reverse if a later stage fails.
        saga_chain: list[tuple[str, str]] = []
        compensations_run: list[str] = []

        async def _step(stage: str) -> None:
            """Execute one pipeline stage; register its compensation on success."""
            await workflow.execute_activity(
                stage,
                order,
                start_to_close_timeout=_TIMEOUT,
                retry_policy=_STEP_RETRY,
            )
            completed.append(stage)
            comp_name = COMPENSATIONS.get(stage)
            if comp_name is not None:
                saga_chain.append((stage, comp_name))

        async def _rollback(aborted_at: str) -> OrderReport:
            """SAGA: unwind the chain in reverse order. refund_payment is
            policy-aware — it's charged a penalty if the confirmation email
            already went out, since that's a broken delivery promise, not
            just a plain cancellation."""
            promise_broken = "send_confirmation_email" in completed
            refund_total: float | None = None

            for stage_name, comp_name in reversed(saga_chain):
                workflow.logger.info(
                    "Compensation", extra={"for": stage_name, "running": comp_name}
                )
                comp_input: object
                if comp_name == "refund_payment":
                    comp_input = RefundInput(
                        order_id=order.order_id,
                        amount=order.amount,
                        penalty_pct=BROKEN_PROMISE_PENALTY_PCT if promise_broken else 0.0,
                    )
                else:
                    comp_input = order.order_id

                comp_json: str = await workflow.execute_activity(
                    comp_name,
                    comp_input,
                    start_to_close_timeout=_TIMEOUT,
                    retry_policy=_COMP_RETRY,
                )
                comp_result = StepResult.model_validate_json(comp_json)
                compensations_run.append(comp_name)
                if comp_result.refund_amount is not None:
                    refund_total = comp_result.refund_amount

            return OrderReport(
                order_id=order.order_id,
                completed_stages=list(completed),
                compensations_run=list(compensations_run),
                succeeded=False,
                aborted_at=aborted_at,
                final_status="cancelled",
                refund_amount=refund_total,
            )

        # ── Stage 1: Payment ─────────────────────────────────────────────────
        try:
            await _step("charge_payment")
        except ActivityError as exc:
            workflow.logger.error("charge_payment failed", extra={"err": str(exc)})
            return OrderReport(
                order_id=order.order_id,
                completed_stages=[],
                compensations_run=[],
                succeeded=False,
                aborted_at="charge_payment",
                final_status="cancelled",
            ).model_dump_json()

        # ── Stage 2: Inventory ───────────────────────────────────────────────
        try:
            await _step("reserve_inventory")
        except ActivityError as exc:
            workflow.logger.error("reserve_inventory failed", extra={"err": str(exc)})
            return (await _rollback("reserve_inventory")).model_dump_json()

        # ── Stage 3: Confirmation email ──────────────────────────────────────
        try:
            await _step("send_confirmation_email")
        except ActivityError as exc:
            workflow.logger.error("send_confirmation_email failed", extra={"err": str(exc)})
            return (await _rollback("send_confirmation_email")).model_dump_json()

        # ── Stage 4: Shipping ────────────────────────────────────────────────
        try:
            await _step("ship_order")
        except ActivityError as exc:
            # SELF-HEAL: the primary warehouse couldn't fulfill — try the
            # backup warehouse once before giving up. This is forward
            # recovery, distinct from the backward-recovery rollback below.
            workflow.logger.warning(
                "ship_order failed — attempting backup warehouse", extra={"err": str(exc)}
            )
            try:
                await _step("ship_order_backup_warehouse")
            except ActivityError as backup_exc:
                # SAGA: this is the failure from the motivating example — no
                # warehouse can fulfill by the promised date, policy says
                # cancel + refund. Since send_confirmation_email already ran,
                # _rollback applies the broken-delivery-promise penalty.
                workflow.logger.error(
                    "Backup warehouse also failed — rolling back",
                    extra={"err": str(backup_exc)},
                )
                return (await _rollback("ship_order")).model_dump_json()

        # ── Success ──────────────────────────────────────────────────────────
        workflow.logger.info("Order fulfilled", extra={"order_id": order.order_id})
        return OrderReport(
            order_id=order.order_id,
            completed_stages=list(completed),
            compensations_run=[],
            succeeded=True,
            aborted_at=None,
            final_status="fulfilled",
        ).model_dump_json()
