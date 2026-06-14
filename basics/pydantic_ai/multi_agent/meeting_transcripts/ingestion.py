"""
Ingestion Layer — pipeline output → PostgreSQL
===============================================

Responsibility: take a completed PipelineOutput and persist it to the three
PostgreSQL tables (meetings, meeting_insights, action_items).

This module has NO knowledge of agents or LLMs. It only reads PipelineOutput
and writes to the database.

Usage (from the CLI or watcher):

    from ingestion import ingest
    await ingest(meeting_id, pipeline_output, db_url)

Schema (idempotent — uses INSERT ... ON CONFLICT DO NOTHING for meetings):

    meetings        (id, title, processed_at, participants, meeting_date)
    meeting_insights(id, meeting_id, insight_type, speaker, content, polarity)
    action_items    (id, meeting_id, owner, action, deadline, verdict, reason)
"""

import logging
from datetime import UTC, datetime

import asyncpg

from pipeline import Insight, PipelineOutput, ValidatedActionItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL (run once via init_schema or manually)
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS meetings (
    id             VARCHAR(64)  PRIMARY KEY,
    title          TEXT         NOT NULL,
    processed_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    participants   TEXT[]       NOT NULL DEFAULT '{}',
    meeting_date   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS meeting_insights (
    id           SERIAL       PRIMARY KEY,
    meeting_id   VARCHAR(64)  NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    insight_type TEXT         NOT NULL
                              CHECK (insight_type IN ('sentiment_shift', 'pain_point', 'competitor')),
    speaker      TEXT,
    content      TEXT         NOT NULL,
    polarity     TEXT,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_mi_meeting ON meeting_insights (meeting_id);
CREATE INDEX IF NOT EXISTS ix_mi_type    ON meeting_insights (insight_type);

CREATE TABLE IF NOT EXISTS action_items (
    id          SERIAL       PRIMARY KEY,
    meeting_id  VARCHAR(64)  NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    owner       TEXT         NOT NULL,
    action      TEXT         NOT NULL,
    deadline    TEXT,
    verdict     TEXT         NOT NULL CHECK (verdict IN ('valid', 'invalid')),
    reason      TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_ai_meeting ON action_items (meeting_id);
CREATE INDEX IF NOT EXISTS ix_ai_owner   ON action_items (owner);
CREATE INDEX IF NOT EXISTS ix_ai_verdict ON action_items (verdict);
"""


async def init_schema(conn: asyncpg.Connection) -> None:
    """Create tables and indexes if they do not already exist."""
    await conn.execute(DDL)
    logger.info("Schema initialised (idempotent)")


# ---------------------------------------------------------------------------
# Ingest helpers
# ---------------------------------------------------------------------------


async def _upsert_meeting(
    conn: asyncpg.Connection,
    meeting_id: str,
    output: PipelineOutput,
    meeting_date: str | None = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO meetings (id, title, processed_at, participants, meeting_date)
        VALUES ($1, $2, $3, $4, $5::timestamptz)
        ON CONFLICT (id) DO UPDATE
            SET title        = EXCLUDED.title,
                processed_at = EXCLUDED.processed_at,
                participants  = EXCLUDED.participants,
                meeting_date  = EXCLUDED.meeting_date
        """,
        meeting_id,
        output.meeting_title,
        datetime.now(UTC),
        output.participants,
        meeting_date,
    )
    logger.debug("Upserted meeting: %s", meeting_id)


async def _insert_insights(
    conn: asyncpg.Connection,
    meeting_id: str,
    insights: Insight,
) -> None:
    rows: list[tuple] = []

    for shift in insights.sentiment_shifts:
        rows.append((
            meeting_id, "sentiment_shift",
            shift.speaker, shift.shift, shift.polarity,
        ))
    for pain in insights.pain_points:
        rows.append((meeting_id, "pain_point", None, pain, None))
    for comp in insights.competitor_mentions:
        rows.append((meeting_id, "competitor", None, comp, None))

    if rows:
        await conn.executemany(
            """
            INSERT INTO meeting_insights
                (meeting_id, insight_type, speaker, content, polarity)
            VALUES ($1, $2, $3, $4, $5)
            """,
            rows,
        )
    logger.debug("Inserted %d insight rows for %s", len(rows), meeting_id)


async def _insert_action_items(
    conn: asyncpg.Connection,
    meeting_id: str,
    items: list[ValidatedActionItem],
) -> None:
    rows = [
        (meeting_id, item.owner, item.action, item.deadline, item.verdict, item.reason)
        for item in items
    ]
    if rows:
        await conn.executemany(
            """
            INSERT INTO action_items
                (meeting_id, owner, action, deadline, verdict, reason)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            rows,
        )
    logger.debug("Inserted %d action item rows for %s", len(rows), meeting_id)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def ingest(
    meeting_id: str,
    output: PipelineOutput,
    db_url: str,
    meeting_date: str | None = None,
) -> None:
    """Persist a PipelineOutput to PostgreSQL.

    Args:
        meeting_id:   The unique meeting identifier (dataset folder name).
        output:       Completed pipeline output.
        db_url:       asyncpg-compatible DSN, e.g.
                      ``postgresql://user:pass@host/dbname``.
        meeting_date: ISO-8601 timestamp of the meeting start (optional).
    """
    conn: asyncpg.Connection = await asyncpg.connect(db_url)
    try:
        async with conn.transaction():
            await _upsert_meeting(conn, meeting_id, output, meeting_date)
            # Clear existing insights/items so re-ingestion is idempotent
            await conn.execute(
                "DELETE FROM meeting_insights WHERE meeting_id = $1", meeting_id
            )
            await conn.execute(
                "DELETE FROM action_items WHERE meeting_id = $1", meeting_id
            )
            await _insert_insights(conn, meeting_id, output.insights)
            await _insert_action_items(conn, meeting_id, output.action_items)
        logger.info(
            "Ingested meeting %s: %d insights, %d action items",
            meeting_id,
            len(output.insights.sentiment_shifts)
            + len(output.insights.pain_points)
            + len(output.insights.competitor_mentions),
            len(output.action_items),
        )
    finally:
        await conn.close()
