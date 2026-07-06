"""Temporal workflow + activities skeleton for the Loan Application Router.

Blueprint companion to ``Design.md`` (see "High-Level Architecture",
"Data Flow & Services", and "Deep Dive 4: Temporal workflow design"). This is a
*skeleton*: every activity body is stubbed with a ``TODO`` and the real I/O is
left out. The value here is the **shape** — decorators, the determinism boundary,
retry policies, the parallel fan-out, and the human-in-the-loop signal/timer.

Determinism boundary (the one rule that matters in Temporal)
------------------------------------------------------------
* **Workflow code is deterministic** and replayed from event history. It may NOT
  do I/O, call ``datetime.now()``/``random``/``uuid4``, spawn threads, or read
  env/config. It orchestrates: calls activities, branches, waits, handles signals.
  Use ``workflow.now()``, ``workflow.logger``, ``workflow.uuid4()`` if needed.
* **Activity code is where all I/O lives** (DB, credit bureau, ID provider,
  Docling, the LLM narrator). Activities can be non-deterministic and are retried
  independently by the engine.

Recommended production split (kept in one file here for readability)::

    models.py       # the Pydantic DTOs
    activities.py   # @activity.defn functions (import I/O clients freely)
    workflow.py     # @workflow.defn — import ONLY activity *references* + models
    worker.py       # Worker bootstrap (registers both)
    client.py       # start_workflow / signal helpers used by the API layer

Temporal loads workflow modules in a sandbox and re-imports them on replay, so
keeping heavy/non-deterministic imports out of ``workflow.py`` matters in prod.

Run locally::

    pip install "temporalio" "pydantic>=2"
    temporal server start-dev                 # Temporal dev server on :7233
    python loan_workflow.py worker            # start the worker
    python loan_workflow.py demo app_123      # start one workflow execution
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

from pydantic import BaseModel, Field

# Temporal SDK. Workflow-safe imports (stdlib, models, pure code) can be imported
# normally. Anything that touches I/O at import time should be wrapped in
# ``with workflow.unsafe.imports_passed_through():`` inside workflow.py.
from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

# ---------------------------------------------------------------------------
# Constants — task queue, routing bands, timeouts, SLA
# ---------------------------------------------------------------------------

TASK_QUEUE = "loan-application"

# Routing bands (deterministic — see Design.md "Why LLM narrator").
AUTO_APPROVE_MIN = 70.0  # score >= 70            -> auto-approve
GRAY_ZONE_MIN = 60.0     # 60 <= score < 70       -> underwriter (HIL)
                         # score < 60             -> deny
HIGH_VALUE_CENTS = 500_000_00  # > $500k always routes to underwriter

# Per-activity timeouts.
IDENTITY_TIMEOUT = timedelta(seconds=30)
CREDIT_TIMEOUT = timedelta(seconds=30)
DOCUMENT_TIMEOUT = timedelta(seconds=60)
RISK_TIMEOUT = timedelta(seconds=120)
NARRATOR_TIMEOUT = timedelta(seconds=30)
PERSIST_TIMEOUT = timedelta(seconds=15)

# Regulatory SLA for a human underwriter decision (Design.md step 6/8).
UNDERWRITER_SLA = timedelta(days=3)

# Default retry policy for external-dependency activities.
EXTERNAL_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,  # after this, the activity fails -> workflow routes to HIL
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Tier(str, Enum):
    AUTO_APPROVE = "auto_approve"
    UNDERWRITER = "underwriter"
    DENY = "deny"


class Verdict(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    PENDING_REVIEW = "pending_review"


# ---------------------------------------------------------------------------
# Data models (DTOs passed between workflow <-> activities)
#
# Payloads are serialized into Temporal's event history, so keep them small and
# free of secrets. Pass the application_id; activities load PII from the DB and
# never put raw SSN into a payload. Use Pydantic + pydantic_data_converter (see
# worker bootstrap) so these serialize automatically.
# ---------------------------------------------------------------------------


class IdentityVerification(BaseModel):
    application_id: str
    status: str                      # verified | mismatch | unavailable
    confidence: float
    provider_ref: str | None = None


class CreditReport(BaseModel):
    application_id: str
    score: int
    dti: float                       # debt-to-income from bureau tradelines
    tradelines: int
    pulled_at: str                   # ISO ts set inside the activity (not workflow)


class IncomeVerification(BaseModel):
    application_id: str
    gross_income_cents: int
    employment_type: str
    confidence: float                # low confidence -> flagged downstream
    verified_at: str


class RiskSignal(BaseModel):
    """Deterministic output of RiskSignalActivity — the routing input.

    ``tier`` is computed here (code, no LLM). The explanation is added later and
    has NO effect on tier/score/routing.
    """

    application_id: str
    score: float
    tier: Tier
    dti: float
    flags: list[str] = Field(default_factory=list)
    regulatory_check_passed: bool = True
    fraud_flag: bool = False
    rule_version: str = "reg-v2.3"


class UnderwriterDecision(BaseModel):
    """Payload delivered via the ``underwriter_decision`` signal."""

    decision: Verdict                       # approved | denied
    reason: str
    conditions: list[str] = Field(default_factory=list)
    adverse_action_reasons: list[str] = Field(default_factory=list)  # required if denied
    decided_by: str                         # "underwriter:u_88" (from JWT, not body)


class Decision(BaseModel):
    """Terminal workflow output — mirrors GET /applications/:id/decision."""

    application_id: str
    tier: Tier
    decision: Verdict
    score: float
    flags: list[str] = Field(default_factory=list)
    explanation: str = ""
    adverse_action_reasons: list[str] = Field(default_factory=list)
    rule_version: str = "reg-v2.3"
    decided_by: str = "system"              # or "underwriter:<id>"


# =====================================================================
# ACTIVITIES  —  all I/O lives here; retried independently by Temporal
# =====================================================================
#
# Each activity takes a small, serializable input and returns a Pydantic model.
# Bodies are stubs. Real implementations: open a DB pool at worker startup and
# reach it via a shared client object / contextvar, wrapped in the reliability
# helpers from the design (timeout + circuit breaker + idempotency).


@activity.defn
async def identity_activity(application_id: str) -> IdentityVerification:
    """Verify stated identity against the ID provider; write IdentityVerification.

    Temporal retries this per EXTERNAL_RETRY on failure. Safe to retry: the ID
    check is read-only against the provider.
    """
    activity.logger.info("identity check", extra={"application_id": application_id})
    # TODO: call ID provider (behind circuit breaker); persist IdentityVerification row.
    raise NotImplementedError


@activity.defn
async def credit_bureau_activity(application_id: str) -> CreditReport:
    """Pull credit score + tradelines. MUST be idempotent — a hard inquiry is a
    paid, FCRA-regulated side effect that must not repeat on retry.

    Idempotency (Design.md Deep Dive 1):
      key = sha256(application_id + "credit_pull")
      1. look up key in processed_ops / Redis
      2. if found -> return the cached CreditReport, DO NOT call the bureau
      3. else -> call bureau, persist result + key atomically, return
    """
    activity.logger.info("credit pull", extra={"application_id": application_id})
    # TODO: idempotency check -> bureau call (breaker) -> persist CreditReport + key.
    raise NotImplementedError


@activity.defn
async def document_activity(application_id: str) -> IncomeVerification:
    """Extract income/employment from uploaded docs via Docling; write
    IncomeVerification. Docling is CPU-bound — run it via asyncio.to_thread so it
    doesn't block the activity's event loop (I/O to object store stays async)."""
    activity.logger.info("document extract", extra={"application_id": application_id})
    # TODO: fetch docs from object store -> await asyncio.to_thread(docling_parse, ...)
    #       -> compare vs stated income -> persist IncomeVerification.
    raise NotImplementedError


@activity.defn
async def risk_signal_activity(
    identity: IdentityVerification,
    credit: CreditReport,
    income: IncomeVerification,
) -> RiskSignal:
    """Compute ALL risk signals deterministically (code, no LLM): DTI, regulatory
    rule check, fraud lookup, anomaly flags, composite score, and the routing
    ``tier``. This is the auditable decision core."""
    activity.logger.info("risk synthesis", extra={"application_id": credit.application_id})
    # TODO: real scoring model + rules-engine query (state, loan_type, effective_date).
    #       Populate flags e.g. "stated_income_mismatch:+24%", "employment_type_conflict".
    raise NotImplementedError


@activity.defn
async def llm_narrator_activity(signal: RiskSignal) -> str:
    """LLM narrator (L1, single call, no tool loop). Turns the structured
    RiskSignal into a human-readable ``explanation`` for the underwriter.

    ADVISORY ONLY — the return value never affects score/tier/routing. Stateless
    and read-only, so safe to retry. Keep a low ``maximum_attempts`` and fall
    back to a templated explanation if the model is unavailable rather than
    stalling the pipeline.
    """
    activity.logger.info("llm narrator", extra={"application_id": signal.application_id})
    # TODO: single Pydantic AI / model call with the serialized signal as context.
    raise NotImplementedError


@activity.defn
async def persist_decision_activity(decision: Decision) -> None:
    """Write the terminal decision to ``risk_decisions`` + append ``audit_log`` in
    one transaction. Idempotent on application_id (unique constraint) so a retry
    after a crash doesn't double-insert."""
    activity.logger.info("persist decision", extra={"application_id": decision.application_id})
    # TODO: INSERT ... ON CONFLICT DO NOTHING; append audit_log row.
    raise NotImplementedError


@activity.defn
async def set_status_activity(application_id: str, status: str) -> None:
    """Update loan_applications.status (pending|processing|awaiting_underwriter|
    decided|withdrawn) so polling GET /applications/:id reflects progress."""
    # TODO: UPDATE loan_applications SET status = $2, updated_at = now() WHERE id = $1.
    raise NotImplementedError


@activity.defn
async def escalate_activity(application_id: str) -> None:
    """SLA breach handler — promote to the senior underwriter queue + audit
    (sla_breach) + page on-call. Fired by the durable timer even after a worker
    restart."""
    activity.logger.warning("SLA breach escalation", extra={"application_id": application_id})
    # TODO: promote to senior queue; audit_log(sla_breach); alert.
    raise NotImplementedError


# =====================================================================
# WORKFLOW  —  deterministic orchestration only (no I/O)
# =====================================================================


@dataclass
class _WorkflowState:
    """In-memory (durably replayed) state for queries + the HIL wait."""

    stage: str = "started"
    human_decision: UnderwriterDecision | None = None


@workflow.defn
class LoanApplicationWorkflow:
    """Durable orchestrator for one loan application (workflow_id = f"loan-{id}").

    Lifecycle: parallel verification -> risk synthesis -> LLM explanation ->
    route (auto-approve / deny complete immediately; gray-zone suspends for a
    human signal with an SLA timer).
    """

    def __init__(self) -> None:
        self._state = _WorkflowState()

    # ---- main entrypoint ------------------------------------------------

    @workflow.run
    async def run(self, application_id: str) -> Decision:
        self._state.stage = "verifying"
        await workflow.execute_activity(
            set_status_activity, args=[application_id, "processing"],
            start_to_close_timeout=PERSIST_TIMEOUT,
        )

        # --- Step 1: three genuinely-independent verifications in parallel ---
        # Each is retried independently; a failure in one does not replay the others.
        identity, credit, income = await asyncio.gather(
            workflow.execute_activity(
                identity_activity, application_id,
                start_to_close_timeout=IDENTITY_TIMEOUT, retry_policy=EXTERNAL_RETRY,
            ),
            workflow.execute_activity(
                credit_bureau_activity, application_id,
                start_to_close_timeout=CREDIT_TIMEOUT, retry_policy=EXTERNAL_RETRY,
            ),
            workflow.execute_activity(
                document_activity, application_id,
                start_to_close_timeout=DOCUMENT_TIMEOUT, retry_policy=EXTERNAL_RETRY,
            ),
            return_exceptions=True,
        )

        # Never auto-decide on missing data: any verification failure -> underwriter.
        failures = [r for r in (identity, credit, income) if isinstance(r, BaseException)]
        if failures:
            workflow.logger.warning("verification incomplete -> underwriter fallback")
            return await self._route_to_underwriter_on_missing_data(application_id)

        assert isinstance(identity, IdentityVerification)
        assert isinstance(credit, CreditReport)
        assert isinstance(income, IncomeVerification)

        # --- Step 2: deterministic risk synthesis (code) ---
        self._state.stage = "risk_synthesis"
        signal = await workflow.execute_activity(
            risk_signal_activity, args=[identity, credit, income],
            start_to_close_timeout=RISK_TIMEOUT,
        )

        # --- Step 3: LLM explanation (advisory; failure is non-fatal) ---
        self._state.stage = "explanation"
        try:
            explanation = await workflow.execute_activity(
                llm_narrator_activity, signal,
                start_to_close_timeout=NARRATOR_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
        except ApplicationError:
            explanation = "Explanation unavailable; see structured risk signals."

        # --- Step 4: route (pure deterministic branching in the workflow) ---
        tier = self._route(signal, application_id_amount_cents=None)  # amount check TODO

        if tier is Tier.AUTO_APPROVE:
            return await self._finalize(
                application_id, signal, explanation, Verdict.APPROVED, decided_by="system",
            )
        if tier is Tier.DENY:
            return await self._finalize(
                application_id, signal, explanation, Verdict.DENIED, decided_by="system",
            )

        # --- Step 5: gray zone -> human-in-the-loop wait + SLA timer ---
        return await self._await_underwriter(application_id, signal, explanation)

    # ---- signal: underwriter submits their decision ---------------------

    @workflow.signal
    def underwriter_decision(self, decision: UnderwriterDecision) -> None:
        """Delivered by the API layer (PUT /underwriter-decision). Only sets
        state; the ``run`` coroutine wakes via wait_condition and does the work.
        Signal handlers must not block or call activities directly."""
        self._state.human_decision = decision

    # ---- query: non-mutating status read (for GET /applications/:id) ----

    @workflow.query
    def current_stage(self) -> str:
        return self._state.stage

    # ---- internal helpers ----------------------------------------------

    def _route(self, signal: RiskSignal, application_id_amount_cents: int | None) -> Tier:
        """Deterministic routing — the auditable decision. No LLM input here."""
        # High-value or hard fraud/regulatory fail always leaves the auto lane.
        if signal.fraud_flag or not signal.regulatory_check_passed:
            return Tier.DENY
        if application_id_amount_cents and application_id_amount_cents > HIGH_VALUE_CENTS:
            return Tier.UNDERWRITER
        if signal.score >= AUTO_APPROVE_MIN:
            return Tier.AUTO_APPROVE
        if signal.score >= GRAY_ZONE_MIN:
            return Tier.UNDERWRITER
        return Tier.DENY

    async def _await_underwriter(
        self, application_id: str, signal: RiskSignal, explanation: str,
    ) -> Decision:
        self._state.stage = "awaiting_underwriter"
        await workflow.execute_activity(
            set_status_activity, args=[application_id, "awaiting_underwriter"],
            start_to_close_timeout=PERSIST_TIMEOUT,
        )

        # Suspend durably until the signal arrives OR the SLA timer fires. This
        # costs no thread — the workflow sleeps in Temporal's persistence layer
        # and the timer survives worker restarts.
        try:
            await workflow.wait_condition(
                lambda: self._state.human_decision is not None,
                timeout=UNDERWRITER_SLA,
            )
        except asyncio.TimeoutError:
            await workflow.execute_activity(
                escalate_activity, application_id,
                start_to_close_timeout=PERSIST_TIMEOUT,
            )
            raise ApplicationError("SLA deadline exceeded", non_retryable=True) from None

        human = self._state.human_decision
        assert human is not None
        return await self._finalize(
            application_id, signal, explanation, human.decision,
            decided_by=human.decided_by,
            adverse_action_reasons=human.adverse_action_reasons,
        )

    async def _route_to_underwriter_on_missing_data(self, application_id: str) -> Decision:
        """Verification failed/exhausted retries — enter HIL with a minimal signal
        rather than fabricating a score."""
        stub = RiskSignal(
            application_id=application_id, score=GRAY_ZONE_MIN, tier=Tier.UNDERWRITER,
            dti=0.0, flags=["verification_incomplete"], regulatory_check_passed=True,
        )
        return await self._await_underwriter(
            application_id, stub, "Verification incomplete — manual review required.",
        )

    async def _finalize(
        self,
        application_id: str,
        signal: RiskSignal,
        explanation: str,
        verdict: Verdict,
        *,
        decided_by: str,
        adverse_action_reasons: list[str] | None = None,
    ) -> Decision:
        decision = Decision(
            application_id=application_id,
            tier=signal.tier,
            decision=verdict,
            score=signal.score,
            flags=signal.flags,
            explanation=explanation,
            adverse_action_reasons=adverse_action_reasons or [],
            rule_version=signal.rule_version,
            decided_by=decided_by,
        )
        await workflow.execute_activity(
            persist_decision_activity, decision,
            start_to_close_timeout=PERSIST_TIMEOUT,
        )
        await workflow.execute_activity(
            set_status_activity, args=[application_id, "decided"],
            start_to_close_timeout=PERSIST_TIMEOUT,
        )
        self._state.stage = "decided"
        return decision


# =====================================================================
# WORKER + CLIENT bootstrap  (would live in worker.py / client.py)
# =====================================================================


async def run_worker() -> None:
    """Start a worker that hosts the workflow + all activities on TASK_QUEUE.

    Workers are stateless and horizontally scalable — run many. Register the
    pydantic data converter so the Pydantic DTOs above serialize automatically.
    """
    from temporalio.client import Client
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.worker import Worker

    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[LoanApplicationWorkflow],
        activities=[
            identity_activity,
            credit_bureau_activity,
            document_activity,
            risk_signal_activity,
            llm_narrator_activity,
            persist_decision_activity,
            set_status_activity,
            escalate_activity,
        ],
    )
    workflow.logger.info("worker starting")  # noqa: F821 (module-level logger stand-in)
    await worker.run()


async def start_application(application_id: str) -> str:
    """API-layer helper: start one workflow execution (Data Flow step 2).

    workflow_id = f"loan-{application_id}" gives dedup for free — starting the
    same id twice is rejected, which backs submission idempotency.
    """
    from temporalio.client import Client
    from temporalio.contrib.pydantic import pydantic_data_converter

    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    handle = await client.start_workflow(
        LoanApplicationWorkflow.run,
        application_id,
        id=f"loan-{application_id}",
        task_queue=TASK_QUEUE,
    )
    return handle.id


async def submit_underwriter_decision(
    application_id: str, decision: UnderwriterDecision,
) -> None:
    """API-layer helper for PUT /underwriter-decision — signal the running
    workflow. Signaling a completed/closed workflow raises (maps to HTTP 409)."""
    from temporalio.client import Client
    from temporalio.contrib.pydantic import pydantic_data_converter

    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    handle = client.get_workflow_handle(f"loan-{application_id}")
    await handle.signal(LoanApplicationWorkflow.underwriter_decision, decision)


# ---------------------------------------------------------------------------
# CLI: `python loan_workflow.py worker` | `... demo <application_id>`
# ---------------------------------------------------------------------------


def _main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "worker"
    if mode == "worker":
        asyncio.run(run_worker())
    elif mode == "demo":
        app_id = sys.argv[2] if len(sys.argv) > 2 else "app_demo"
        asyncio.run(start_application(app_id))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    _main()
