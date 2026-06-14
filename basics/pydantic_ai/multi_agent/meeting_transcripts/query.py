"""
Query Layer — read insights from PostgreSQL
===========================================

Responsibility: provide typed async query functions for all stakeholder
questions defined in test_questions.md.

This module has NO knowledge of agents or LLMs. It only reads from the
tables created by ingestion.py.

Usage:

    from query import connect, get_action_items_for_owner, get_pain_points
    conn = await connect(db_url)
    items = await get_action_items_for_owner(conn, owner="Raj Kapoor")
"""

import logging
from dataclasses import dataclass

import asyncpg

logger = logging.getLogger(__name__)


async def connect(db_url: str) -> asyncpg.Connection:
    return await asyncpg.connect(db_url)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class MeetingRow:
    id: str
    title: str
    processed_at: str
    participants: list[str]
    meeting_date: str | None


@dataclass
class InsightRow:
    meeting_id: str
    meeting_title: str
    insight_type: str
    speaker: str | None
    content: str
    polarity: str | None
    meeting_date: str | None


@dataclass
class ActionItemRow:
    meeting_id: str
    meeting_title: str
    owner: str
    action: str
    deadline: str | None
    verdict: str
    reason: str | None
    meeting_date: str | None


@dataclass
class HallucinationSummary:
    meeting_title: str
    valid_items: int
    invalid_items: int
    hallucination_pct: float


@dataclass
class SentimentOwnerRow:
    speaker: str
    negative: int
    mixed: int
    positive: int
    neutral: int


# ---------------------------------------------------------------------------
# Head-of-Engineering queries
# ---------------------------------------------------------------------------


async def get_incident_meetings(conn: asyncpg.Connection) -> list[InsightRow]:
    """Q1 — Meetings with active infrastructure incidents."""
    rows = await conn.fetch(
        """
        SELECT m.id, m.title, m.meeting_date, mi.insight_type, mi.speaker,
               mi.content, mi.polarity
        FROM meetings m
        JOIN meeting_insights mi ON mi.meeting_id = m.id
        WHERE mi.insight_type = 'pain_point'
          AND mi.content ILIKE ANY (ARRAY[
              '%outage%', '%incident%', '%downtime%', '%failure%', '%error%'
          ])
        ORDER BY m.meeting_date DESC NULLS LAST
        """
    )
    return [
        InsightRow(
            meeting_id=r["id"], meeting_title=r["title"],
            insight_type=r["insight_type"], speaker=r["speaker"],
            content=r["content"], polarity=r["polarity"],
            meeting_date=str(r["meeting_date"]) if r["meeting_date"] else None,
        )
        for r in rows
    ]


async def get_open_action_items(
    conn: asyncpg.Connection,
    verdict: str = "valid",
) -> list[ActionItemRow]:
    """Q2 — Open (valid) action items with deadlines."""
    rows = await conn.fetch(
        """
        SELECT ai.owner, ai.action, ai.deadline, ai.verdict, ai.reason,
               m.id AS meeting_id, m.title, m.meeting_date
        FROM action_items ai
        JOIN meetings m ON m.id = ai.meeting_id
        WHERE ai.verdict = $1
          AND ai.deadline IS NOT NULL
          AND ai.deadline <> 'Unspecified'
        ORDER BY ai.owner, ai.deadline
        """,
        verdict,
    )
    return [
        ActionItemRow(
            meeting_id=r["meeting_id"], meeting_title=r["title"],
            owner=r["owner"], action=r["action"], deadline=r["deadline"],
            verdict=r["verdict"], reason=r["reason"],
            meeting_date=str(r["meeting_date"]) if r["meeting_date"] else None,
        )
        for r in rows
    ]


async def get_top_pain_points(
    conn: asyncpg.Connection, limit: int = 10
) -> list[tuple[str, int]]:
    """Q3 — Most common pain points across all meetings."""
    rows = await conn.fetch(
        """
        SELECT mi.content, COUNT(DISTINCT mi.meeting_id) AS meeting_count
        FROM meeting_insights mi
        WHERE mi.insight_type = 'pain_point'
        GROUP BY mi.content
        ORDER BY meeting_count DESC, mi.content
        LIMIT $1
        """,
        limit,
    )
    return [(r["content"], r["meeting_count"]) for r in rows]


async def get_hallucination_summary(conn: asyncpg.Connection) -> list[HallucinationSummary]:
    """Q5 — Hallucination rate (invalid %) per meeting."""
    rows = await conn.fetch(
        """
        SELECT m.title,
               COUNT(*) FILTER (WHERE ai.verdict = 'valid')   AS valid_items,
               COUNT(*) FILTER (WHERE ai.verdict = 'invalid') AS invalid_items,
               ROUND(
                 COUNT(*) FILTER (WHERE ai.verdict = 'invalid')::numeric /
                 NULLIF(COUNT(*), 0) * 100, 1
               ) AS hallucination_pct
        FROM meetings m
        JOIN action_items ai ON ai.meeting_id = m.id
        GROUP BY m.id, m.title
        ORDER BY hallucination_pct DESC
        """
    )
    return [
        HallucinationSummary(
            meeting_title=r["title"],
            valid_items=r["valid_items"],
            invalid_items=r["invalid_items"],
            hallucination_pct=float(r["hallucination_pct"] or 0),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Product queries
# ---------------------------------------------------------------------------


async def get_positive_sentiment_shifts(
    conn: asyncpg.Connection,
) -> list[InsightRow]:
    """Q6 — Positively framed sentiment shifts (features mentioned well)."""
    rows = await conn.fetch(
        """
        SELECT m.id, m.title, m.meeting_date, mi.insight_type,
               mi.speaker, mi.content, mi.polarity
        FROM meeting_insights mi
        JOIN meetings m ON m.id = mi.meeting_id
        WHERE mi.insight_type = 'sentiment_shift'
          AND mi.polarity IN ('positive', 'mixed')
        ORDER BY m.meeting_date DESC NULLS LAST
        """
    )
    return [
        InsightRow(
            meeting_id=r["id"], meeting_title=r["title"],
            insight_type=r["insight_type"], speaker=r["speaker"],
            content=r["content"], polarity=r["polarity"],
            meeting_date=str(r["meeting_date"]) if r["meeting_date"] else None,
        )
        for r in rows
    ]


async def get_competitor_mentions(conn: asyncpg.Connection) -> list[InsightRow]:
    """Q7 — All competitor mentions across meetings."""
    rows = await conn.fetch(
        """
        SELECT m.id, m.title, m.meeting_date, mi.insight_type,
               mi.speaker, mi.content, mi.polarity
        FROM meeting_insights mi
        JOIN meetings m ON m.id = mi.meeting_id
        WHERE mi.insight_type = 'competitor'
        ORDER BY m.meeting_date DESC NULLS LAST
        """
    )
    return [
        InsightRow(
            meeting_id=r["id"], meeting_title=r["title"],
            insight_type=r["insight_type"], speaker=r["speaker"],
            content=r["content"], polarity=None,
            meeting_date=str(r["meeting_date"]) if r["meeting_date"] else None,
        )
        for r in rows
    ]


async def get_sentiment_by_speaker(conn: asyncpg.Connection) -> list[SentimentOwnerRow]:
    """Q10 — Sentiment breakdown by speaker."""
    rows = await conn.fetch(
        """
        SELECT mi.speaker,
               COUNT(*) FILTER (WHERE mi.polarity = 'negative') AS negative,
               COUNT(*) FILTER (WHERE mi.polarity = 'mixed')    AS mixed,
               COUNT(*) FILTER (WHERE mi.polarity = 'positive') AS positive,
               COUNT(*) FILTER (WHERE mi.polarity = 'neutral')  AS neutral
        FROM meeting_insights mi
        WHERE mi.insight_type = 'sentiment_shift'
          AND mi.speaker IS NOT NULL
        GROUP BY mi.speaker
        ORDER BY negative DESC, mixed DESC
        """
    )
    return [
        SentimentOwnerRow(
            speaker=r["speaker"],
            negative=r["negative"], mixed=r["mixed"],
            positive=r["positive"], neutral=r["neutral"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Customer Support queries
# ---------------------------------------------------------------------------


async def get_customer_facing_commitments(conn: asyncpg.Connection) -> list[ActionItemRow]:
    """Q11 — Action items explicitly related to customer communication."""
    rows = await conn.fetch(
        """
        SELECT ai.owner, ai.action, ai.deadline, ai.verdict, ai.reason,
               m.id AS meeting_id, m.title, m.meeting_date
        FROM action_items ai
        JOIN meetings m ON m.id = ai.meeting_id
        WHERE ai.verdict = 'valid'
          AND (
              ai.action ILIKE '%customer%'
              OR ai.action ILIKE '%communication%'
              OR ai.action ILIKE '%email%'
              OR ai.action ILIKE '%notify%'
              OR ai.action ILIKE '%update%'
          )
        ORDER BY m.meeting_date DESC NULLS LAST
        """
    )
    return [
        ActionItemRow(
            meeting_id=r["meeting_id"], meeting_title=r["title"],
            owner=r["owner"], action=r["action"], deadline=r["deadline"],
            verdict=r["verdict"], reason=r["reason"],
            meeting_date=str(r["meeting_date"]) if r["meeting_date"] else None,
        )
        for r in rows
    ]


async def get_negative_sentiment(conn: asyncpg.Connection) -> list[InsightRow]:
    """Q13 — What are customers most frustrated about?"""
    rows = await conn.fetch(
        """
        SELECT m.id, m.title, m.meeting_date, mi.insight_type,
               mi.speaker, mi.content, mi.polarity
        FROM meeting_insights mi
        JOIN meetings m ON m.id = mi.meeting_id
        WHERE mi.insight_type = 'sentiment_shift'
          AND mi.polarity = 'negative'
        ORDER BY m.meeting_date DESC NULLS LAST
        """
    )
    return [
        InsightRow(
            meeting_id=r["id"], meeting_title=r["title"],
            insight_type=r["insight_type"], speaker=r["speaker"],
            content=r["content"], polarity=r["polarity"],
            meeting_date=str(r["meeting_date"]) if r["meeting_date"] else None,
        )
        for r in rows
    ]


async def get_action_items_for_owner(
    conn: asyncpg.Connection, owner: str
) -> list[ActionItemRow]:
    """Q14 — Action items for a specific owner (supports partial match)."""
    rows = await conn.fetch(
        """
        SELECT ai.owner, ai.action, ai.deadline, ai.verdict, ai.reason,
               m.id AS meeting_id, m.title, m.meeting_date
        FROM action_items ai
        JOIN meetings m ON m.id = ai.meeting_id
        WHERE ai.verdict = 'valid'
          AND ai.owner ILIKE $1
        ORDER BY ai.deadline
        """,
        f"%{owner}%",
    )
    return [
        ActionItemRow(
            meeting_id=r["meeting_id"], meeting_title=r["title"],
            owner=r["owner"], action=r["action"], deadline=r["deadline"],
            verdict=r["verdict"], reason=r["reason"],
            meeting_date=str(r["meeting_date"]) if r["meeting_date"] else None,
        )
        for r in rows
    ]


async def get_meetings_with_no_actions(conn: asyncpg.Connection) -> list[MeetingRow]:
    """Q15 — Meetings that produced zero valid action items."""
    rows = await conn.fetch(
        """
        SELECT m.id, m.title, m.processed_at, m.participants, m.meeting_date
        FROM meetings m
        WHERE NOT EXISTS (
            SELECT 1 FROM action_items ai
            WHERE ai.meeting_id = m.id AND ai.verdict = 'valid'
        )
        ORDER BY m.meeting_date DESC NULLS LAST
        """
    )
    return [
        MeetingRow(
            id=r["id"], title=r["title"],
            processed_at=str(r["processed_at"]),
            participants=list(r["participants"]),
            meeting_date=str(r["meeting_date"]) if r["meeting_date"] else None,
        )
        for r in rows
    ]
