"""OrderWorkflow — deterministic orchestration (no I/O; all side effects via activities).

Flow: processing -> validate -> (if high-value: await human approval + SLA timer)
-> confirmed / rejected. The high-value branch is the human-in-the-loop pattern:
the workflow suspends durably on `wait_condition` until a signal arrives or the SLA
timer fires.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from app import domain
    from app.domain import Decision, OrderStatus
    from app.models import Order, OrderResult
    from app.temporal.activities import OrderActivities

APPROVAL_SLA = timedelta(days=2)          # regulatory-style deadline for a human decision
ACTIVITY_TIMEOUT = timedelta(seconds=10)
RETRY = RetryPolicy(maximum_attempts=3)


@workflow.defn
class OrderWorkflow:
    def __init__(self) -> None:
        self._decision: Decision | None = None

    @workflow.run
    async def run(self, order_id: str) -> OrderResult:
        await self._set(order_id, OrderStatus.PROCESSING)

        order: Order = await workflow.execute_activity(
            OrderActivities.load_order, args=[order_id],
            start_to_close_timeout=ACTIVITY_TIMEOUT, retry_policy=RETRY,
        )

        if domain.validate_order(order.item, order.quantity, order.unit_price_cents):
            return await self._reject(order_id)

        if domain.is_high_value(order.total_cents):
            await self._set(order_id, OrderStatus.AWAITING_APPROVAL)
            # Suspend durably: no thread held. Resumes on signal, or the SLA timer fires.
            try:
                await workflow.wait_condition(
                    lambda: self._decision is not None, timeout=APPROVAL_SLA
                )
            except TimeoutError:
                return await self._reject(order_id)  # SLA breach -> auto-reject
            if self._decision is Decision.REJECTED:
                return await self._reject(order_id)

        await self._set(order_id, OrderStatus.CONFIRMED)
        return OrderResult(order_id=order_id, status=OrderStatus.CONFIRMED)

    @workflow.signal
    def submit_approval(self, decision: Decision) -> None:
        # Signal handlers only mutate state; the run() coroutine does the work.
        self._decision = decision

    @workflow.query
    def decision_pending(self) -> bool:
        return self._decision is None

    # -- helpers --

    async def _set(self, order_id: str, status: OrderStatus) -> None:
        await workflow.execute_activity(
            OrderActivities.mark_status, args=[order_id, status],
            start_to_close_timeout=ACTIVITY_TIMEOUT, retry_policy=RETRY,
        )

    async def _reject(self, order_id: str) -> OrderResult:
        await self._set(order_id, OrderStatus.REJECTED)
        return OrderResult(order_id=order_id, status=OrderStatus.REJECTED)
