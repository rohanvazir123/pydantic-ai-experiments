"""Scheduled ingestion job store — CRUD against the scheduled_jobs table."""

import logging
import uuid as _uuid
from datetime import UTC, datetime
from typing import Any, cast

import asyncpg
from croniter import croniter

from knowledge.config.settings import Settings, load_settings

logger = logging.getLogger(__name__)


def compute_next_run_at(cron_expr: str, base: datetime | None = None) -> datetime:
    """Return the next fire time for a cron expression (UTC)."""
    it = croniter(cron_expr, base or datetime.now(UTC))
    return cast("datetime", it.get_next(datetime))


class ScheduledJobStore:
    """CRUD for the scheduled_jobs PostgreSQL table."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_settings()
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._settings.database_url,
            min_size=1,
            max_size=3,
            command_timeout=self._settings.db_query_timeout_s,
        )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def create(
        self,
        tenant_id: str,
        name: str,
        source_type: str,
        source_config: dict[str, str],
        corpus_id: str,
        cron_expr: str,
        mode: str = "incremental",
        enable_graph_extraction: bool = False,
    ) -> str:
        import json
        assert self._pool
        job_id     = str(_uuid.uuid4())
        next_run   = compute_next_run_at(cron_expr)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO scheduled_jobs
                  (id, tenant_id, name, source_type, source_config,
                   corpus_id, cron_expr, mode, enable_graph_extraction, next_run_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                """,
                job_id, tenant_id, name, source_type, json.dumps(source_config),
                corpus_id, cron_expr, mode, enable_graph_extraction, next_run,
            )
        return job_id

    async def get(self, job_id: str, tenant_id: str) -> dict[str, Any] | None:
        assert self._pool
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM scheduled_jobs WHERE id=$1 AND tenant_id=$2",
                job_id, tenant_id,
            )
        return dict(row) if row else None

    async def list_by_tenant(self, tenant_id: str) -> list[dict[str, Any]]:
        assert self._pool
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM scheduled_jobs WHERE tenant_id=$1 ORDER BY created_at DESC",
                tenant_id,
            )
        return [dict(r) for r in rows]

    async def get_due_jobs(self) -> list[dict[str, Any]]:
        """Return all active jobs whose next_run_at <= NOW()."""
        assert self._pool
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM scheduled_jobs
                WHERE is_active = TRUE AND next_run_at <= NOW()
                ORDER BY next_run_at ASC
                """,
            )
        return [dict(r) for r in rows]

    async def update_next_run_at(self, job_id: str, cron_expr: str) -> None:
        assert self._pool
        next_run = compute_next_run_at(cron_expr)
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE scheduled_jobs SET next_run_at=$1, updated_at=NOW() WHERE id=$2",
                next_run, job_id,
            )

    async def update_last_run(self, job_id: str, status: str, ingest_job_id: str) -> None:
        assert self._pool
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE scheduled_jobs
                SET last_run_at=NOW(), last_status=$1, last_job_id=$2, updated_at=NOW()
                WHERE id=$3
                """,
                status, ingest_job_id, job_id,
            )

    async def delete(self, job_id: str, tenant_id: str) -> None:
        assert self._pool
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM scheduled_jobs WHERE id=$1 AND tenant_id=$2",
                job_id, tenant_id,
            )
