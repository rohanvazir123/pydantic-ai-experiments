"""
Load Transcript Intelligence JSON files into a requirement-oriented Postgres schema.

Unlike load_dataset_to_postgres.py, this script does not preserve every JSON field.
It inspects the dataset for supported source files, creates only analytical tables
needed for the take-home requirements, and discards redundant/noisy source data.

Functional requirements covered:
  1. Categorize transcripts by topic/theme
  2. Analyze sentiment across inferred call types
  3. Surface additional signals such as churn, concerns, feature gaps, and action items

Usage:
    python basics/iprep/i1/load_functional_schema_to_postgres.py --reset
    python basics/iprep/i1/load_functional_schema_to_postgres.py --dry-run

Connection environment variables:
    PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset"
DEFAULT_SCHEMA = "iprep_i1_functional"
DEFAULT_TAXONOMY_PATH = SCRIPT_DIR / "taxonomy.json"

SUPPORTED_FILES = {
    "meeting-info",
    "summary",
    "transcript",
}

DISCARDED_FILES = {
    "speakers": "redundant with transcript_lines speaker/timing fields",
    "events": "join/leave telemetry is not needed for the stated analysis",
    "speaker-meta": "redundant with transcript_lines speaker_id/speaker_name",
}

THEME_KEYWORDS = {
    "Reliability / Incidents / Outages": [
        "outage",
        "incident",
        "sla",
        "failure",
        "latency",
        "degradation",
        "reliability",
        "post-mortem",
        "postmortem",
        "single point",
        "recovery",
    ],
    "Compliance / Audit / Security Assurance": [
        "compliance",
        "audit",
        "soc 2",
        "hipaa",
        "pci",
        "gdpr",
        "cmmc",
        "regulatory",
        "evidence",
    ],
    "Identity / Access / Security Controls": [
        "identity",
        "access",
        "mfa",
        "sso",
        "scim",
        "okta",
        "active directory",
        "rbac",
        "provisioning",
        "saml",
        "ldap",
        "session timeout",
    ],
    "Product Feedback / Feature Gaps / Roadmap": [
        "feature",
        "roadmap",
        "product",
        "early access",
        "launch",
        "feedback",
        "custom reporting",
        "pdf",
        "ux",
    ],
    "Customer Retention / Renewal / Commercial Risk": [
        "renewal",
        "churn",
        "pricing",
        "contract",
        "billing",
        "credit",
        "competitive",
        "upsell",
        "expansion",
        "retention",
    ],
    "Support / Customer Escalation": [
        "support",
        "escalation",
        "ticket",
        "workaround",
        "bug",
        "customer communication",
        "response time",
    ],
    "Implementation / Onboarding / Adoption": [
        "onboarding",
        "deployment",
        "migration",
        "integration",
        "adoption",
        "connector",
        "configuration",
        "rollout",
    ],
    "Internal Engineering / Planning / Execution": [
        "sprint",
        "qa",
        "technical debt",
        "tech debt",
        "architecture",
        "planning",
        "design review",
        "capacity",
        "resource allocation",
    ],
}

CALL_TYPE_HINTS = {
    "support_escalation": [
        "support",
        "ticket",
        "escalation",
        "bug",
        "workaround",
        "response time",
        "outage impact",
    ],
    "sales_or_renewal": [
        "renewal",
        "pricing",
        "contract",
        "billing",
        "upsell",
        "expansion",
        "qbr",
        "competitive",
    ],
    "internal_incident": [
        "post-mortem",
        "postmortem",
        "incident review",
        "incident response",
        "root cause",
        "remediation",
    ],
    "internal_planning": [
        "sprint",
        "qa",
        "roadmap planning",
        "architecture",
        "design review",
        "capacity planning",
    ],
    "external_customer": [
        "onboarding",
        "adoption",
        "customer feedback",
        "product demo",
        "proof of concept",
        "implementation",
    ],
}


@dataclass
class MeetingRecord:
    meeting_id: str
    meeting_info: dict[str, Any]
    summary: dict[str, Any]
    transcript_lines: list[dict[str, Any]]


@dataclass
class Taxonomy:
    theme_keywords: dict[str, list[str]]
    call_type_hints: dict[str, list[str]]


def load_dotenv() -> None:
    for env_file in (SCRIPT_DIR.parents[2] / ".env", SCRIPT_DIR / ".env"):
        if not env_file.exists():
            continue
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() and key.strip() not in os.environ:
                os.environ[key.strip()] = value.strip().strip('"').strip("'")


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


def clean_topic(topic: str) -> str:
    return re.sub(r"\s+", " ", topic.strip().lower())


def normalize_keywords(values: list[Any]) -> list[str]:
    return sorted({clean_topic(str(value)) for value in values if str(value).strip()})


def load_taxonomy(path: Path | None) -> Taxonomy:
    if not path or not path.exists():
        return Taxonomy(THEME_KEYWORDS, CALL_TYPE_HINTS)

    payload = json.loads(path.read_text(encoding="utf-8"))
    themes = payload.get("themes", [])
    call_types = payload.get("call_types", [])

    theme_keywords = {
        item["theme_name"]: normalize_keywords(item.get("keyword_hints", []))
        for item in themes
        if item.get("theme_name") and item.get("keyword_hints")
    }
    call_type_hints = {
        item["call_type"]: normalize_keywords(item.get("keyword_hints", []))
        for item in call_types
        if item.get("call_type") and item.get("keyword_hints")
    }

    if not theme_keywords:
        raise ValueError(f"No themes with keyword_hints found in {path}")
    if not call_type_hints:
        raise ValueError(f"No call_types with keyword_hints found in {path}")

    return Taxonomy(theme_keywords, call_type_hints)


def parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return None


def score_keywords(text: str, keyword_map: dict[str, list[str]]) -> Counter[str]:
    normalized = clean_topic(text)
    scores: Counter[str] = Counter()
    for label, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in normalized:
                scores[label] += 1
    return scores


def infer_themes(
    topics: list[str],
    key_moment_types: list[str],
    summary: str,
    taxonomy: Taxonomy,
) -> Counter[str]:
    scores: Counter[str] = Counter()
    for topic in topics:
        scores.update(score_keywords(topic, taxonomy.theme_keywords))

    if "technical_issue" in key_moment_types:
        scores["Reliability / Incidents / Outages"] += 1
        scores["Support / Customer Escalation"] += 1
    if "churn_signal" in key_moment_types:
        scores["Customer Retention / Renewal / Commercial Risk"] += 2
    if "feature_gap" in key_moment_types:
        scores["Product Feedback / Feature Gaps / Roadmap"] += 2
    if "concern" in key_moment_types:
        scores.update(score_keywords(summary, taxonomy.theme_keywords))

    return scores


def infer_call_type(
    topics: list[str],
    summary: str,
    theme_scores: Counter[str],
    taxonomy: Taxonomy,
) -> tuple[str, float]:
    scores: Counter[str] = Counter()
    combined = " ".join(topics + [summary])
    scores.update(score_keywords(combined, taxonomy.call_type_hints))

    if theme_scores["Customer Retention / Renewal / Commercial Risk"] > 0:
        scores["sales_or_renewal"] += 1
    if theme_scores["Support / Customer Escalation"] > 0:
        scores["support_escalation"] += 1
    if theme_scores["Internal Engineering / Planning / Execution"] > 0:
        scores["internal_planning"] += 1
    if theme_scores["Reliability / Incidents / Outages"] > 0 and "customer" not in clean_topic(combined):
        scores["internal_incident"] += 1

    if not scores:
        return "unknown", 0.0

    call_type, score = scores.most_common(1)[0]
    confidence = min(0.95, 0.45 + (score * 0.15))
    return call_type, confidence


def load_records(dataset_dir: Path) -> tuple[list[MeetingRecord], Counter[str]]:
    records: list[MeetingRecord] = []
    file_counts: Counter[str] = Counter()

    for meeting_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        payloads: dict[str, Any] = {}
        for json_file in sorted(meeting_dir.glob("*.json")):
            file_type = json_file.stem
            file_counts[file_type] += 1
            if file_type in SUPPORTED_FILES:
                payloads[file_type] = json.loads(json_file.read_text(encoding="utf-8"))

        records.append(
            MeetingRecord(
                meeting_id=meeting_dir.name,
                meeting_info=payloads.get("meeting-info", {}),
                summary=payloads.get("summary", {}),
                transcript_lines=payloads.get("transcript", {}).get("data", []),
            )
        )

    return records, file_counts


async def create_schema(conn: asyncpg.Connection, schema: str, reset: bool) -> None:
    q_schema = quote_ident(schema)
    if reset:
        await conn.execute(f"DROP SCHEMA IF EXISTS {q_schema} CASCADE")
    await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {q_schema}")

    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {q_schema}.meetings (
            meeting_id text PRIMARY KEY,
            title text,
            organizer_email text,
            host text,
            start_time timestamptz,
            end_time timestamptz,
            duration_minutes numeric
        )
        """
    )
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {q_schema}.meeting_participants (
            meeting_id text REFERENCES {q_schema}.meetings(meeting_id),
            email text NOT NULL,
            participant_role text NOT NULL,
            PRIMARY KEY (meeting_id, email, participant_role)
        )
        """
    )
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {q_schema}.meeting_summaries (
            meeting_id text PRIMARY KEY REFERENCES {q_schema}.meetings(meeting_id),
            summary text,
            overall_sentiment text,
            sentiment_score numeric
        )
        """
    )
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {q_schema}.summary_topics (
            meeting_id text REFERENCES {q_schema}.meetings(meeting_id),
            topic text NOT NULL,
            normalized_topic text NOT NULL,
            PRIMARY KEY (meeting_id, normalized_topic)
        )
        """
    )
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {q_schema}.action_items (
            meeting_id text REFERENCES {q_schema}.meetings(meeting_id),
            action_index int NOT NULL,
            owner text,
            action_item text NOT NULL,
            PRIMARY KEY (meeting_id, action_index)
        )
        """
    )
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {q_schema}.key_moments (
            meeting_id text REFERENCES {q_schema}.meetings(meeting_id),
            moment_index int NOT NULL,
            time_seconds numeric,
            speaker_name text,
            moment_type text,
            text text,
            PRIMARY KEY (meeting_id, moment_index)
        )
        """
    )
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {q_schema}.transcript_lines (
            meeting_id text REFERENCES {q_schema}.meetings(meeting_id),
            line_index int NOT NULL,
            speaker_id int,
            speaker_name text,
            sentiment_type text,
            start_seconds numeric,
            end_seconds numeric,
            confidence numeric,
            sentence text,
            PRIMARY KEY (meeting_id, line_index)
        )
        """
    )
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {q_schema}.meeting_themes (
            meeting_id text REFERENCES {q_schema}.meetings(meeting_id),
            theme text NOT NULL,
            evidence_count int NOT NULL,
            is_primary boolean NOT NULL DEFAULT false,
            PRIMARY KEY (meeting_id, theme)
        )
        """
    )
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {q_schema}.call_types (
            meeting_id text PRIMARY KEY REFERENCES {q_schema}.meetings(meeting_id),
            call_type text NOT NULL,
            confidence numeric NOT NULL
        )
        """
    )
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {q_schema}.sentiment_features (
            meeting_id text PRIMARY KEY REFERENCES {q_schema}.meetings(meeting_id),
            total_lines int NOT NULL,
            positive_lines int NOT NULL,
            neutral_lines int NOT NULL,
            negative_lines int NOT NULL,
            positive_ratio numeric NOT NULL,
            negative_ratio numeric NOT NULL,
            net_sentiment numeric NOT NULL,
            concern_count int NOT NULL,
            churn_signal_count int NOT NULL,
            technical_issue_count int NOT NULL,
            feature_gap_count int NOT NULL,
            praise_count int NOT NULL,
            action_item_count int NOT NULL
        )
        """
    )

    for table in (
        "summary_topics",
        "key_moments",
        "transcript_lines",
        "meeting_themes",
        "call_types",
    ):
        await conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_meeting_idx ON {q_schema}.{table}(meeting_id)")


async def truncate_tables(conn: asyncpg.Connection, schema: str) -> None:
    q_schema = quote_ident(schema)
    tables = [
        "sentiment_features",
        "call_types",
        "meeting_themes",
        "transcript_lines",
        "key_moments",
        "action_items",
        "summary_topics",
        "meeting_summaries",
        "meeting_participants",
        "meetings",
    ]
    await conn.execute(
        "TRUNCATE "
        + ", ".join(f"{q_schema}.{table}" for table in tables)
        + " RESTART IDENTITY CASCADE"
    )


def split_action_owner(action_item: str) -> tuple[str | None, str]:
    owner, sep, rest = action_item.partition(":")
    if sep and len(owner) <= 80:
        return owner.strip(), rest.strip()
    return None, action_item.strip()


async def load_functional_tables(
    conn: asyncpg.Connection,
    schema: str,
    records: list[MeetingRecord],
    taxonomy: Taxonomy,
) -> None:
    q_schema = quote_ident(schema)

    for record in records:
        info = record.meeting_info
        summary = record.summary
        meeting_id = summary.get("meetingId") or info.get("meetingId") or record.meeting_id

        await conn.execute(
            f"""
            INSERT INTO {q_schema}.meetings
            VALUES ($1, $2, $3, $4, $5::timestamptz, $6::timestamptz, $7)
            """,
            meeting_id,
            info.get("title"),
            info.get("organizerEmail"),
            info.get("host"),
            parse_timestamp(info.get("startTime")),
            parse_timestamp(info.get("endTime")),
            info.get("duration"),
        )

        for role, emails in (("all_email", info.get("allEmails", [])), ("invitee", info.get("invitees", []))):
            for email in emails:
                await conn.execute(
                    f"INSERT INTO {q_schema}.meeting_participants VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
                    meeting_id,
                    email,
                    role,
                )

        await conn.execute(
            f"INSERT INTO {q_schema}.meeting_summaries VALUES ($1, $2, $3, $4)",
            meeting_id,
            summary.get("summary"),
            summary.get("overallSentiment"),
            summary.get("sentimentScore"),
        )

        topics = [str(topic) for topic in summary.get("topics", [])]
        for topic in topics:
            await conn.execute(
                f"INSERT INTO {q_schema}.summary_topics VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
                meeting_id,
                topic,
                clean_topic(topic),
            )

        for index, action_item in enumerate(summary.get("actionItems", []), start=1):
            owner, item_text = split_action_owner(str(action_item))
            await conn.execute(
                f"INSERT INTO {q_schema}.action_items VALUES ($1, $2, $3, $4)",
                meeting_id,
                index,
                owner,
                item_text,
            )

        key_moment_types: list[str] = []
        key_moment_counts: Counter[str] = Counter()
        for index, moment in enumerate(summary.get("keyMoments", []), start=1):
            moment_type = moment.get("type")
            if moment_type:
                key_moment_types.append(moment_type)
                key_moment_counts[moment_type] += 1
            await conn.execute(
                f"INSERT INTO {q_schema}.key_moments VALUES ($1, $2, $3, $4, $5, $6)",
                meeting_id,
                index,
                moment.get("time"),
                moment.get("speaker"),
                moment_type,
                moment.get("text"),
            )

        sentiment_counts: Counter[str] = Counter()
        for line in record.transcript_lines:
            sentiment_type = line.get("sentimentType") or "unknown"
            sentiment_counts[sentiment_type] += 1
            await conn.execute(
                f"""
                INSERT INTO {q_schema}.transcript_lines
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                meeting_id,
                line.get("index"),
                line.get("speaker_id"),
                line.get("speaker_name"),
                sentiment_type,
                line.get("time"),
                line.get("endTime"),
                line.get("averageConfidence"),
                line.get("sentence"),
            )

        theme_scores = infer_themes(topics, key_moment_types, summary.get("summary") or "", taxonomy)
        primary_theme = theme_scores.most_common(1)[0][0] if theme_scores else None
        for theme, evidence_count in theme_scores.items():
            await conn.execute(
                f"INSERT INTO {q_schema}.meeting_themes VALUES ($1, $2, $3, $4)",
                meeting_id,
                theme,
                evidence_count,
                theme == primary_theme,
            )

        call_type, confidence = infer_call_type(
            topics,
            summary.get("summary") or "",
            theme_scores,
            taxonomy,
        )
        await conn.execute(
            f"INSERT INTO {q_schema}.call_types VALUES ($1, $2, $3)",
            meeting_id,
            call_type,
            confidence,
        )

        total_lines = sum(sentiment_counts.values())
        positive_lines = sentiment_counts["positive"]
        neutral_lines = sentiment_counts["neutral"]
        negative_lines = sentiment_counts["negative"]
        positive_ratio = positive_lines / total_lines if total_lines else 0.0
        negative_ratio = negative_lines / total_lines if total_lines else 0.0
        net_sentiment = positive_ratio - negative_ratio
        await conn.execute(
            f"""
            INSERT INTO {q_schema}.sentiment_features
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """,
            meeting_id,
            total_lines,
            positive_lines,
            neutral_lines,
            negative_lines,
            positive_ratio,
            negative_ratio,
            net_sentiment,
            key_moment_counts["concern"],
            key_moment_counts["churn_signal"],
            key_moment_counts["technical_issue"],
            key_moment_counts["feature_gap"],
            key_moment_counts["praise"],
            len(summary.get("actionItems", [])),
        )


def print_schema_plan(file_counts: Counter[str], records: list[MeetingRecord]) -> None:
    print("Detected source files:")
    for file_type, count in sorted(file_counts.items()):
        action = "load" if file_type in SUPPORTED_FILES else "discard"
        reason = DISCARDED_FILES.get(file_type, "unsupported/no requirement mapping")
        suffix = "" if action == "load" else f" ({reason})"
        print(f"  {file_type:14s} {count:4d}  -> {action}{suffix}")

    print("\nGenerated analytical tables:")
    for table in (
        "meetings",
        "meeting_participants",
        "meeting_summaries",
        "summary_topics",
        "action_items",
        "key_moments",
        "transcript_lines",
        "meeting_themes",
        "call_types",
        "sentiment_features",
    ):
        print(f"  {table}")

    topic_counts = Counter()
    moment_counts = Counter()
    for record in records:
        topic_counts.update(clean_topic(str(t)) for t in record.summary.get("topics", []))
        moment_counts.update(m.get("type") for m in record.summary.get("keyMoments", []) if m.get("type"))

    print("\nTop detected topics:")
    for topic, count in topic_counts.most_common(12):
        print(f"  {topic:35s} {count:4d}")

    print("\nDetected key moment signals:")
    for moment_type, count in moment_counts.most_common():
        print(f"  {moment_type:20s} {count:4d}")


async def print_counts(conn: asyncpg.Connection, schema: str) -> None:
    q_schema = quote_ident(schema)
    tables = [
        "meetings",
        "meeting_summaries",
        "summary_topics",
        "action_items",
        "key_moments",
        "transcript_lines",
        "meeting_themes",
        "call_types",
        "sentiment_features",
    ]
    print("\nLoaded row counts:")
    for table in tables:
        count = await conn.fetchval(f"SELECT count(*) FROM {q_schema}.{table}")
        print(f"  {table:20s} {count:>8,}")

    print("\nUseful DBeaver queries:")
    print(f"  SELECT * FROM {schema}.meeting_themes WHERE is_primary ORDER BY theme, evidence_count DESC;")
    print(f"  SELECT call_type, count(*), avg(negative_ratio) FROM {schema}.call_types JOIN {schema}.sentiment_features USING (meeting_id) GROUP BY 1;")
    print(f"  SELECT theme, count(*) FROM {schema}.meeting_themes GROUP BY 1 ORDER BY 2 DESC;")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY_PATH,
        help="Taxonomy JSON from the LLM. Falls back to built-in rules if missing.",
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show inferred schema plan without connecting to Postgres.")
    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset folder not found: {dataset_dir}")

    records, file_counts = load_records(dataset_dir)
    taxonomy = load_taxonomy(args.taxonomy)
    print_schema_plan(file_counts, records)
    print(f"\nTaxonomy source: {args.taxonomy if args.taxonomy.exists() else 'built-in fallback'}")
    print(f"  themes:     {len(taxonomy.theme_keywords)}")
    print(f"  call types: {len(taxonomy.call_type_hints)}")

    if args.dry_run:
        return

    try:
        import asyncpg  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install asyncpg first: pip install asyncpg") from exc

    print(f"\nConnecting to {HOST}:{PORT}/{DATABASE} as {USER}")
    conn = await asyncpg.connect(dsn())
    try:
        await create_schema(conn, args.schema, args.reset)
        await truncate_tables(conn, args.schema)
        await load_functional_tables(conn, args.schema, records, taxonomy)
        await print_counts(conn, args.schema)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
