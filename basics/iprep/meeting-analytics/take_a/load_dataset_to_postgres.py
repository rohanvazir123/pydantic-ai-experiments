"""
Load the Transcript Intelligence JSON dataset into Postgres for DBeaver browsing.

This keeps the initial database shape intentionally simple:
  - one raw JSONB table with one row per JSON file
  - a few SQL views that flatten the obvious things worth scanning

Usage:
    python basics/iprep/meeting-analytics/load_dataset_to_postgres.py
    python basics/iprep/meeting-analytics/load_dataset_to_postgres.py --reset
    python basics/iprep/meeting-analytics/load_dataset_to_postgres.py --schema transcript_intel

Connection environment variables:
    PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import asyncpg


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR.parent / "dataset"
DEFAULT_SCHEMA = "iprep_meeting-analytics"


def load_dotenv() -> None:
    """Load .env from repo root or this folder without requiring python-dotenv."""
    for env_file in (SCRIPT_DIR.parent / ".env",):
        if not env_file.exists():
            continue
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv()

HOST = os.getenv("PG_HOST", "localhost")
PORT = int(os.getenv("PG_PORT", "5432"))
USER = os.getenv("PG_USER", "postgres")
PASSWORD = os.getenv("PG_PASSWORD", "")
DATABASE = os.getenv("PG_DATABASE", "postgres")


def dsn() -> str:
    if PASSWORD:
        return f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
    return f"postgresql://{USER}@{HOST}:{PORT}/{DATABASE}"


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def normalize_file_type(path: Path) -> str:
    return path.stem


def iter_json_files(dataset_dir: Path) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    for meeting_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for json_file in sorted(meeting_dir.glob("*.json")):
            rows.append((meeting_dir.name, normalize_file_type(json_file), json_file))
    return rows


async def create_schema_objects(
    conn: asyncpg.Connection, schema: str, reset: bool
) -> None:
    q_schema = quote_ident(schema)

    if reset:
        await conn.execute(f"DROP SCHEMA IF EXISTS {q_schema} CASCADE")

    await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {q_schema}")
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {q_schema}.raw_files (
            meeting_id text NOT NULL,
            file_type text NOT NULL,
            payload jsonb NOT NULL,
            source_path text NOT NULL,
            loaded_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (meeting_id, file_type)
        )
        """
    )
    await conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS raw_files_file_type_idx
        ON {q_schema}.raw_files (file_type)
        """
    )
    await conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS raw_files_payload_gin_idx
        ON {q_schema}.raw_files
        USING gin (payload)
        """
    )

    await conn.execute(
        f"""
        CREATE OR REPLACE VIEW {q_schema}.meetings AS
        SELECT
            meeting_id,
            payload->>'title' AS title,
            payload->>'organizerEmail' AS organizer_email,
            payload->>'host' AS host,
            (payload->>'startTime')::timestamptz AS start_time,
            (payload->>'endTime')::timestamptz AS end_time,
            (payload->>'duration')::numeric AS duration_minutes,
            payload->'allEmails' AS all_emails,
            payload->'invitees' AS invitees
        FROM {q_schema}.raw_files
        WHERE file_type = 'meeting-info'
        """
    )

    await conn.execute(
        f"""
        CREATE OR REPLACE VIEW {q_schema}.transcript_lines AS
        SELECT
            r.meeting_id,
            (line->>'index')::int AS line_index,
            line->>'speaker_name' AS speaker_name,
            line->>'speaker_id' AS speaker_id,
            line->>'sentimentType' AS sentiment_type,
            (line->>'time')::numeric AS start_seconds,
            (line->>'endTime')::numeric AS end_seconds,
            (line->>'averageConfidence')::numeric AS confidence,
            line->>'sentence' AS sentence
        FROM {q_schema}.raw_files r
        CROSS JOIN LATERAL jsonb_array_elements(r.payload->'data') AS line
        WHERE r.file_type = 'transcript'
        """
    )

    await conn.execute(
        f"""
        CREATE OR REPLACE VIEW {q_schema}.summaries AS
        SELECT
            meeting_id,
            payload->>'summary' AS summary,
            payload->>'overallSentiment' AS overall_sentiment,
            (payload->>'sentimentScore')::numeric AS sentiment_score,
            payload->'topics' AS topics,
            payload->'actionItems' AS action_items,
            payload->'keyMoments' AS key_moments
        FROM {q_schema}.raw_files
        WHERE file_type = 'summary'
        """
    )

    await conn.execute(
        f"""
        CREATE OR REPLACE VIEW {q_schema}.summary_topics AS
        SELECT
            r.meeting_id,
            topic.value #>> '{{}}' AS topic
        FROM {q_schema}.raw_files r
        CROSS JOIN LATERAL jsonb_array_elements(r.payload->'topics') AS topic
        WHERE r.file_type = 'summary'
        """
    )

    await conn.execute(
        f"""
        CREATE OR REPLACE VIEW {q_schema}.action_items AS
        SELECT
            r.meeting_id,
            item.ordinality::int AS action_index,
            item.value #>> '{{}}' AS action_item
        FROM {q_schema}.raw_files r
        CROSS JOIN LATERAL jsonb_array_elements(r.payload->'actionItems')
            WITH ORDINALITY AS item(value, ordinality)
        WHERE r.file_type = 'summary'
        """
    )

    await conn.execute(
        f"""
        CREATE OR REPLACE VIEW {q_schema}.speaker_turns AS
        SELECT
            r.meeting_id,
            turn.ordinality::int - 1 AS turn_index,
            turn.value->>'speakerName' AS speaker_name,
            (turn.value->>'timestamp')::numeric AS start_seconds,
            (turn.value->>'endTimeTs')::numeric AS end_seconds
        FROM {q_schema}.raw_files r
        CROSS JOIN LATERAL jsonb_array_elements(r.payload)
            WITH ORDINALITY AS turn(value, ordinality)
        WHERE r.file_type = 'speakers'
        """
    )

    await conn.execute(
        f"""
        CREATE OR REPLACE VIEW {q_schema}.participant_events AS
        SELECT
            r.meeting_id,
            event.ordinality::int - 1 AS event_index,
            event.value->>'participantName' AS participant_name,
            event.value->>'type' AS event_type,
            (event.value->>'time')::numeric AS seconds_from_start,
            to_timestamp((event.value->>'timestamp')::numeric / 1000.0) AS event_time
        FROM {q_schema}.raw_files r
        CROSS JOIN LATERAL jsonb_array_elements(r.payload)
            WITH ORDINALITY AS event(value, ordinality)
        WHERE r.file_type = 'events'
        """
    )

    await conn.execute(
        f"""
        CREATE OR REPLACE VIEW {q_schema}.speaker_meta AS
        SELECT
            r.meeting_id,
            meta.key::int AS speaker_id,
            meta.value AS speaker_name
        FROM {q_schema}.raw_files r
        CROSS JOIN LATERAL jsonb_each_text(r.payload) AS meta(key, value)
        WHERE r.file_type = 'speaker-meta'
        """
    )


async def load_dataset(conn: asyncpg.Connection, schema: str, dataset_dir: Path) -> int:
    rows = iter_json_files(dataset_dir)
    q_schema = quote_ident(schema)

    for meeting_id, file_type, path in rows:
        payload = json.loads(path.read_text(encoding="utf-8"))
        await conn.execute(
            f"""
            INSERT INTO {q_schema}.raw_files (meeting_id, file_type, payload, source_path, loaded_at)
            VALUES ($1, $2, $3::jsonb, $4, now())
            ON CONFLICT (meeting_id, file_type)
            DO UPDATE SET
                payload = EXCLUDED.payload,
                source_path = EXCLUDED.source_path,
                loaded_at = now()
            """,
            meeting_id,
            file_type,
            json.dumps(payload),
            str(path),
        )

    return len(rows)


async def print_counts(conn: asyncpg.Connection, schema: str) -> None:
    q_schema = quote_ident(schema)
    raw_count = await conn.fetchval(f"SELECT count(*) FROM {q_schema}.raw_files")
    meeting_count = await conn.fetchval(f"SELECT count(*) FROM {q_schema}.meetings")
    line_count = await conn.fetchval(
        f"SELECT count(*) FROM {q_schema}.transcript_lines"
    )
    summary_count = await conn.fetchval(f"SELECT count(*) FROM {q_schema}.summaries")
    speaker_turn_count = await conn.fetchval(
        f"SELECT count(*) FROM {q_schema}.speaker_turns"
    )
    event_count = await conn.fetchval(
        f"SELECT count(*) FROM {q_schema}.participant_events"
    )

    print("\nLoaded:")
    print(f"  raw_files         {raw_count:>8,}")
    print(f"  meetings          {meeting_count:>8,}")
    print(f"  transcript_lines  {line_count:>8,}")
    print(f"  summaries         {summary_count:>8,}")
    print(f"  speaker_turns     {speaker_turn_count:>8,}")
    print(f"  participant_events {event_count:>7,}")

    print("\nTry these in DBeaver:")
    print(f"  SELECT * FROM {schema}.meetings LIMIT 20;")
    print(
        f"  SELECT * FROM {schema}.transcript_lines ORDER BY meeting_id, line_index LIMIT 100;"
    )
    print(
        f"  SELECT topic, count(*) FROM {schema}.summary_topics GROUP BY 1 ORDER BY 2 DESC;"
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset folder. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
        help=f"Postgres schema to create/use. Default: {DEFAULT_SCHEMA}",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the target schema before loading.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset folder not found: {dataset_dir}")

    print(f"Connecting to {HOST}:{PORT}/{DATABASE} as {USER}")
    print(f"Dataset: {dataset_dir}")
    print(f"Schema:  {args.schema}")

    conn = await asyncpg.connect(dsn())
    try:
        await create_schema_objects(conn, args.schema, args.reset)
        inserted = await load_dataset(conn, args.schema, dataset_dir)
        print(f"\nRead {inserted:,} JSON files from disk.")
        await print_counts(conn, args.schema)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
