# Temporal Python SDK — Reference

## Core decorators

```python
@workflow.defn          # marks the class as a workflow definition
class MyWorkflow:

    @workflow.run       # entry point — exactly one per workflow class
    async def run(self, input: str) -> str:
        ...

    @workflow.signal    # inbound signal handler (write — no return value)
    def my_signal(self, value: str) -> None:
        ...

    @workflow.query     # inbound query handler (read-only — must not mutate state)
    def my_query(self) -> str:
        ...
```

Activities are plain async functions — all real I/O (DB, HTTP, LLM calls) happens here:

```python
@activity.defn
async def credit_bureau_activity(application_id: str) -> CreditReport:
    # real network call lives here, not in the workflow
    ...
```

---

## Workflow vs Activity — the key constraint

| | Workflow | Activity |
|---|---|---|
| Deterministic? | **Must be** — no I/O, no randomness | No constraint |
| `datetime.now()` | Use `workflow.now()` instead | Fine |
| Network / DB calls | Not allowed directly | Where all I/O lives |
| Retried on failure? | Replayed from event history | Yes, with RetryPolicy |
| Runs in | Workflow worker (sandboxed) | Activity worker (normal) |

---

## Executing activities from a workflow

```python
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

result = await workflow.execute_activity(
    my_activity,
    arg,
    start_to_close_timeout=timedelta(seconds=30),
    retry_policy=RetryPolicy(
        maximum_attempts=3,
        initial_interval=timedelta(seconds=1),
        backoff_coefficient=2.0,
    ),
)
```

---

## Parallel activities

```python
results = await asyncio.gather(
    workflow.execute_activity(identity_activity, app_id,
        start_to_close_timeout=timedelta(seconds=30)),
    workflow.execute_activity(credit_bureau_activity, app_id,
        start_to_close_timeout=timedelta(seconds=30)),
    workflow.execute_activity(document_activity, app_id,
        start_to_close_timeout=timedelta(seconds=60)),
)
identity, credit, docs = results
```

---

## Human-in-the-loop (HIL) — signal + timer pattern

```python
@workflow.defn
class LoanApplicationWorkflow:

    def __init__(self) -> None:
        self._human_decision: Decision | None = None

    @workflow.run
    async def run(self, application_id: str) -> Decision:
        # ... run activities ...

        if decision.tier == "gray_zone":
            # suspend and wait for signal OR deadline
            deadline_reached = False

            async def wait_for_decision() -> bool:
                return self._human_decision is not None

            try:
                await workflow.wait_condition(
                    wait_for_decision,
                    timeout=timedelta(days=3),   # regulatory SLA
                )
            except asyncio.TimeoutError:
                deadline_reached = True

            if deadline_reached:
                await workflow.execute_activity(escalate_activity, application_id,
                    start_to_close_timeout=timedelta(seconds=10))
                raise ApplicationError("SLA deadline exceeded — escalated")

            return self._human_decision

        return decision

    @workflow.signal
    def underwriter_decision(self, decision: Decision) -> None:
        self._human_decision = decision
```

Sending the signal from application code:

```python
handle = client.get_workflow_handle(application_id)
await handle.signal(LoanApplicationWorkflow.underwriter_decision, decision)
```

---

## Starting a workflow

```python
from temporalio.client import Client

client = await Client.connect("localhost:7233")

handle = await client.start_workflow(
    LoanApplicationWorkflow.run,
    application_id,
    id=application_id,          # workflow ID = application_id for dedup
    task_queue="loan-processing",
)
```

---

## Running workers

```python
from temporalio.worker import Worker

worker = Worker(
    client,
    task_queue="loan-processing",
    workflows=[LoanApplicationWorkflow],
    activities=[
        identity_activity,
        credit_bureau_activity,
        document_activity,
        risk_synthesis_activity,
        escalate_activity,
    ],
)
await worker.run()
```

---

## Key timeout types

| Timeout | Scope | Use for |
|---------|-------|---------|
| `start_to_close_timeout` | Single activity attempt | How long one try can take |
| `schedule_to_close_timeout` | All attempts combined | Hard cap across all retries |
| `schedule_to_start_timeout` | Time waiting in queue | Detect backed-up workers |
| `workflow.wait_condition(..., timeout=)` | Workflow-level wait | HIL deadline, SLA timer |

---

## When to use Temporal

- Multi-step pipeline where each step can fail independently and must not restart from scratch
- Human-in-the-loop waits that span minutes to days
- Regulatory SLA deadlines that must fire automatically even after worker restarts
- Paid side effects (credit pulls, charges) that must not repeat on retry
- Long-running workflows where a simple queue + worker would require custom state management, polling, and cron jobs to replicate the same guarantees
