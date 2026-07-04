"""
Run the IncidentResponseWorkflow against a real Temporal server + Ollama.

Prerequisites:
    1. Temporal server running:   temporal server start-dev
    2. Ollama running:            ollama serve
    3. Model pulled:              ollama pull qwen2.5:14b

Usage:
    # from repo root
    uv run python basics/workflows/incident_response/run_live.py

    # use a different model tier
    AGENT_LARGE_MODEL=qwen2.5:7b uv run python basics/workflows/incident_response/run_live.py

    # run worker only (in one terminal) then trigger in another
    uv run python basics/workflows/incident_response/run_live.py --worker-only
"""
from __future__ import annotations

import asyncio
import sys

from temporalio.client import Client
from temporalio.worker import Worker

from ..config import get_model
from .activities import InfraActivities, LLMActivities
from .models import IncidentAlert, IncidentReport
from .workflows import IncidentResponseWorkflow

TASK_QUEUE = "incident-response"

SAMPLE_ALERT = IncidentAlert(
    alert_id="INC-2024-001",
    service="payment-service",
    error_rate=0.45,
    latency_p99_ms=3200,
    description="Spike in 5xx errors and p99 latency after the 14:30 deployment. "
                "DB connection pool exhaustion suspected.",
)


async def run_worker(client: Client) -> None:
    infra = InfraActivities()
    llm = LLMActivities(model=get_model("large"))

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[IncidentResponseWorkflow],
        activities=[
            infra.restart_service,
            infra.scale_up,
            infra.scale_down,
            infra.clear_cache,
            infra.rollback_deployment,
            infra.page_oncall,
            llm.triage_incident,
            llm.assess_after_action,
        ],
    )
    print(f"[worker] started on task queue '{TASK_QUEUE}' — qwen2.5:14b via Ollama")
    await worker.run()


async def run_workflow(client: Client) -> None:
    report_json = await client.execute_workflow(
        IncidentResponseWorkflow.run,
        SAMPLE_ALERT.model_dump_json(),
        id=SAMPLE_ALERT.alert_id,
        task_queue=TASK_QUEUE,
    )
    report = IncidentReport.model_validate_json(report_json)
    print("\n── Incident Report ──────────────────────────────────")
    print(f"  Alert:     {report.alert_id}")
    print(f"  Severity:  {report.severity}")
    print(f"  Resolved:  {report.resolved}")
    print(f"  Escalated: {report.escalated}")
    print(f"  Status:    {report.final_status}")
    print(f"  Actions:   {[a.action for a in report.actions_taken]}")
    if report.compensations:
        print(f"  Compensations: {report.compensations}")
    print("─────────────────────────────────────────────────────\n")


async def main() -> None:
    client = await Client.connect("localhost:7233")
    worker_only = "--worker-only" in sys.argv

    if worker_only:
        await run_worker(client)
    else:
        # Run worker + workflow in the same process for the demo
        infra = InfraActivities()
        llm = LLMActivities(model=get_model("large"))
        worker = Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[IncidentResponseWorkflow],
            activities=[
                infra.restart_service,
                infra.scale_up,
                infra.scale_down,
                infra.clear_cache,
                infra.rollback_deployment,
                infra.page_oncall,
                llm.triage_incident,
                llm.assess_after_action,
            ],
        )
        async with worker:
            await run_workflow(client)


if __name__ == "__main__":
    asyncio.run(main())
