"""Production ASGI entrypoint — the app gunicorn/uvicorn serves.

    gunicorn app.api.asgi:app -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000

On startup (FastAPI lifespan) it connects Postgres + Temporal and stores the repo
and workflow starter on `app.state`; the routes read them from there.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.app import create_app
from app.config import get_settings
from app.store.postgres import PostgresOrderRepository
from app.temporal.client import TemporalWorkflowStarter, connect


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    repo = PostgresOrderRepository(settings.database_url)
    await repo.connect()
    client = await connect(settings.temporal_target)
    app.state.repo = repo
    app.state.starter = TemporalWorkflowStarter(client, settings.task_queue)
    try:
        yield
    finally:
        await repo.close()


app = create_app(lifespan=_lifespan)
