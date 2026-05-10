"""
Load raw dataset JSON into base Postgres tables.

Self-contained — reads from dataset/ JSON only, no other modules needed.
All tables are created in the meeting_analytics schema.

Why these tables exist:
  The insight queries in load_output_csvs_to_db.py join key_moments for churn/feature-gap
  signals, meeting_summaries for sentiment, and transcript_lines for
  speaker-level sentiment. These are the data substrate — separate from the
  semantic clustering tables (semantic_clusters, semantic_phrases,
  semantic_meeting_themes) which are written by load_output_csvs_to_db.py.

Tables created:
  meetings              meeting_id, title, organizer_email, duration_minutes, start_time
  meeting_participants  meeting_id, email
  meeting_summaries     meeting_id, summary_text, overall_sentiment, sentiment_score
  key_moments           meeting_id, moment_index, moment_type, text, speaker, time_seconds
  action_items          meeting_id, item_index, owner, text
  transcript_lines      meeting_id, line_index, speaker, sentence, sentiment_type, time_seconds

Usage (from repo root):
  python basics/iprep/meeting-analytics/final_version/load_raw_jsons_to_db.py
  python basics/iprep/meeting-analytics/final_version/load_raw_jsons_to_db.py --reset
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import asyncpg

SCHEMA = "meeting_analytics"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR.parent / "dataset"


def _add_dataset_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)


def _load_dotenv() -> None:
    env_file = SCRIPT_DIR.parent / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip():
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


_load_dotenv()


def _build_dsn() -> str:
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "")
    database = os.getenv("PG_DATABASE", "postgres")
    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return f"postgresql://{user}@{host}:{port}/{database}"


async def drop_schema(conn: asyncpg.Connection) -> None:
    """Drop the entire meeting_analytics schema (including all tables)."""
    await conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
    print(f"  Dropped schema {SCHEMA}")


async def create_base_tables(conn: asyncpg.Connection) -> None:
    """Create the 6 base tables for raw meeting data."""
    await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.meetings (
            meeting_id        TEXT PRIMARY KEY,
            title             TEXT,
            organizer_email   TEXT,
            duration_minutes  NUMERIC,
            start_time        TIMESTAMPTZ
        )
    """)

    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.meeting_participants (
            meeting_id  TEXT NOT NULL
                            REFERENCES {SCHEMA}.meetings(meeting_id) ON DELETE CASCADE,
            email       TEXT NOT NULL,
            PRIMARY KEY (meeting_id, email)
        )
    """)

    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.meeting_summaries (
            meeting_id         TEXT PRIMARY KEY
                                   REFERENCES {SCHEMA}.meetings(meeting_id) ON DELETE CASCADE,
            summary_text       TEXT,
            overall_sentiment  TEXT,
            sentiment_score    NUMERIC,
            topics             TEXT[]   DEFAULT '{{}}'
        )
    """)

    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.key_moments (
            meeting_id    TEXT NOT NULL
                              REFERENCES {SCHEMA}.meetings(meeting_id) ON DELETE CASCADE,
            moment_index  INTEGER NOT NULL,
            moment_type   TEXT,
            text          TEXT,
            speaker       TEXT,
            time_seconds  NUMERIC,
            PRIMARY KEY (meeting_id, moment_index)
        )
    """)

    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.action_items (
            meeting_id  TEXT NOT NULL
                            REFERENCES {SCHEMA}.meetings(meeting_id) ON DELETE CASCADE,
            item_index  INTEGER NOT NULL,
            owner       TEXT,
            text        TEXT,
            PRIMARY KEY (meeting_id, item_index)
        )
    """)

    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.transcript_lines (
            meeting_id     TEXT NOT NULL
                               REFERENCES {SCHEMA}.meetings(meeting_id) ON DELETE CASCADE,
            line_index     INTEGER NOT NULL,
            speaker        TEXT,
            sentence       TEXT,
            sentiment_type TEXT,
            time_seconds   NUMERIC,
            PRIMARY KEY (meeting_id, line_index)
        )
    """)

    await conn.execute(f"""
        CREATE INDEX IF NOT EXISTS key_moments_type_idx
        ON {SCHEMA}.key_moments (meeting_id, moment_type)
    """)
    await conn.execute(f"""
        CREATE INDEX IF NOT EXISTS transcript_lines_meeting_idx
        ON {SCHEMA}.transcript_lines (meeting_id)
    """)


def _parse_dt(raw: str | None) -> datetime | None:
    """Parse an ISO-8601 string (including Z suffix) into an aware datetime."""
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _parse_action_item(raw: str) -> tuple[str, str]:
    """Split 'Owner Name: description text' into (owner, text)."""
    if ":" in raw:
        owner, _, text = raw.partition(":")
        return owner.strip(), text.strip()
    return "", raw.strip()


async def load_files_into_db(
    conn: asyncpg.Connection, dataset_dir: Path
) -> dict[str, int]:
    """Read all raw JSON files from dataset_dir and insert into the 6 base tables."""
    meeting_rows: list[tuple] = []
    participant_rows: list[tuple] = []
    summary_rows: list[tuple] = []
    key_moment_rows: list[tuple] = []
    action_item_rows: list[tuple] = []
    transcript_rows: list[tuple] = []

    for meeting_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        info_path = meeting_dir / "meeting-info.json"
        if not info_path.exists():
            continue

        info = json.loads(info_path.read_text(encoding="utf-8"))
        meeting_id = str(info.get("meetingId") or meeting_dir.name)
        meeting_rows.append((
            meeting_id,
            info.get("title", ""),
            info.get("organizerEmail") or info.get("host", ""),
            info.get("duration"),
            _parse_dt(info.get("startTime")),
        ))

        for email in info.get("allEmails", []):
            if email:
                participant_rows.append((meeting_id, email))

        summary_path = meeting_dir / "summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_rows.append((
                meeting_id,
                payload.get("summary", ""),
                payload.get("overallSentiment"),
                payload.get("sentimentScore"),
                [str(t) for t in payload.get("topics", [])],
            ))

            for idx, km in enumerate(payload.get("keyMoments", [])):
                key_moment_rows.append((
                    meeting_id,
                    idx,
                    km.get("type"),
                    km.get("text", ""),
                    km.get("speaker", ""),
                    km.get("time"),
                ))

            for idx, raw_item in enumerate(payload.get("actionItems", [])):
                owner, text = _parse_action_item(str(raw_item))
                action_item_rows.append((meeting_id, idx, owner, text))

        transcript_path = meeting_dir / "transcript.json"
        if transcript_path.exists():
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            for line in transcript.get("data", []):
                transcript_rows.append((
                    meeting_id,
                    int(line.get("index", 0)),
                    line.get("speaker_name", ""),
                    line.get("sentence", ""),
                    line.get("sentimentType", ""),
                    line.get("time"),
                ))

    await conn.executemany(
        f"""INSERT INTO {SCHEMA}.meetings
                (meeting_id, title, organizer_email, duration_minutes, start_time)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (meeting_id) DO NOTHING""",
        meeting_rows,
    )
    await conn.executemany(
        f"""INSERT INTO {SCHEMA}.meeting_participants (meeting_id, email)
            VALUES ($1, $2)
            ON CONFLICT DO NOTHING""",
        participant_rows,
    )
    await conn.executemany(
        f"""INSERT INTO {SCHEMA}.meeting_summaries
                (meeting_id, summary_text, overall_sentiment, sentiment_score, topics)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (meeting_id) DO NOTHING""",
        summary_rows,
    )
    await conn.executemany(
        f"""INSERT INTO {SCHEMA}.key_moments
                (meeting_id, moment_index, moment_type, text, speaker, time_seconds)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT DO NOTHING""",
        key_moment_rows,
    )
    await conn.executemany(
        f"""INSERT INTO {SCHEMA}.action_items
                (meeting_id, item_index, owner, text)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT DO NOTHING""",
        action_item_rows,
    )
    await conn.executemany(
        f"""INSERT INTO {SCHEMA}.transcript_lines
                (meeting_id, line_index, speaker, sentence, sentiment_type, time_seconds)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT DO NOTHING""",
        transcript_rows,
    )

    return {
        "meetings": len(meeting_rows),
        "participants": len(participant_rows),
        "summaries": len(summary_rows),
        "key_moments": len(key_moment_rows),
        "action_items": len(action_item_rows),
        "transcript_lines": len(transcript_rows),
    }


async def setup(
    dsn: str | None = None,
    dataset_dir: Path | None = None,
    reset: bool = False,
) -> dict[str, int]:
    """
    Create base tables and load raw dataset JSON into Postgres.

    Args:
        dsn:         Postgres DSN. Reads from .env if None.
        dataset_dir: Path to the dataset/ folder. Defaults to ../dataset/.
        reset:       Drop and recreate the entire schema first.

    Returns:
        Row counts per table.
    """
    dsn = dsn or _build_dsn()
    dataset_dir = (dataset_dir or SCRIPT_DIR.parent / "dataset").resolve()

    conn = await asyncpg.connect(dsn)
    try:
        if reset:
            await drop_schema(conn)
        await create_base_tables(conn)
        counts = await load_files_into_db(conn, dataset_dir)
    finally:
        await conn.close()

    return counts


async def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Load raw dataset JSON into base Postgres tables."
    )
    _add_dataset_arg(parser)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the entire schema first.",
    )
    args = parser.parse_args()

    print(f"Loading base data from {args.dataset}")
    if args.reset:
        print("  --reset: dropping schema first")

    counts = await setup(dataset_dir=args.dataset, reset=args.reset)
    for table, n in counts.items():
        print(f"  {table:<20s}: {n}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(_cli())
