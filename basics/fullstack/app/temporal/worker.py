"""Temporal worker entrypoint — hosts OrderWorkflow + activities against Postgres."""

from __future__ import annotations

import asyncio

from temporalio.worker import Worker

from app.config import get_settings
from app.store.postgres import PostgresOrderRepository
from app.temporal.activities import OrderActivities
from app.temporal.client import connect
from app.temporal.workflow import OrderWorkflow


async def main() -> None:
    settings = get_settings()
    repo = PostgresOrderRepository(settings.database_url)
    await repo.connect()
    client = await connect(settings.temporal_target)
    activities = OrderActivities(repo)
    worker = Worker(
        client,
        task_queue=settings.task_queue,
        workflows=[OrderWorkflow],
        activities=[activities.load_order, activities.mark_status],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
