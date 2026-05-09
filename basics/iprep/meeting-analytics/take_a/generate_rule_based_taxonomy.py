"""
Generate and apply a rule-based functional taxonomy for Transcript Intelligence.

This script is intentionally *not* a generic JSON-to-table loader. The
functional requirements ask for categories, sentiment trends, and business insights, so the
database schema is shaped around those analytical requirements:

1. Preserve fields that explain what a meeting was about:
   meeting metadata, summaries, topic tags, action items, key moments, and
   transcript lines.
2. Derive requirement-specific features:
   meeting themes, inferred call type, sentiment ratios, and risk/opportunity
   signal counts.
3. Discard source files that are redundant or low-value for this analysis:
   speaker join/leave telemetry and duplicate speaker metadata are not loaded.

Why this does not need an LLM:
The dataset already contains structured analytical fields that an LLM would
normally be asked to extract from raw text:

- summary.json topics describe what each meeting is about.
- summary.json keyMoments[].type flags business signals such as churn_signal,
  concern, technical_issue, feature_gap, praise, and pricing_offer.
- summary.json overallSentiment and sentimentScore give meeting-level
  sentiment.
- transcript.json data[].sentimentType gives sentence-level sentiment evidence.

Because those fields already exist, the problem is not open-ended text
understanding. It is normalization: mapping many granular topic tags into a
smaller set of stakeholder-useful themes, then computing repeatable sentiment
and risk features. A deterministic ruleset is a better first choice because it
is auditable, cheap to rerun, easy to debug in SQL, and simple to explain during
Q&A.

Rule-based taxonomy approach:
The script supports two taxonomy sources, both deterministic at runtime:

- Built-in deterministic fallback rules:
  THEME_KEYWORDS and CALL_TYPE_HINTS are hand-reviewed keyword maps built from
  the observed summary topic vocabulary and the assignment's stakeholder needs.

- External taxonomy JSON:
  If basics/iprep/meeting-analytics/taxonomy.json exists, load_taxonomy() reads human-reviewed
  taxonomy config and uses that instead of the fallback maps.
  The expected JSON shape is:

      {
        "themes": [
          {
            "theme_name": "Reliability / Incidents / Outages",
            "keyword_hints": ["outage", "incident", "sla"]
          }
        ],
        "call_types": [
          {
            "call_type": "support_escalation",
            "keyword_hints": ["support", "ticket", "escalation"]
          }
        ]
      }

The important design choice is that classification stays deterministic. An LLM
could optionally help brainstorm taxonomy config, but this pipeline does not
require one and does not call one. It only consumes explicit keyword config and
produces repeatable database rows.

Usage:
    python basics/iprep/meeting-analytics/generate_rule_based_taxonomy.py --dry-run
    python basics/iprep/meeting-analytics/generate_rule_based_taxonomy.py --reset
    python basics/iprep/meeting-analytics/generate_rule_based_taxonomy.py --taxonomy basics/iprep/meeting-analytics/taxonomy.json --reset

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
DEFAULT_DATASET_DIR = SCRIPT_DIR.parent / "dataset"
DEFAULT_SCHEMA = "meeting_analytics"
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
    """
    In-memory representation of one meeting folder from the dataset.

    The source dataset stores each meeting as a directory containing multiple
    JSON files. This class gathers only the files needed for the functional
    requirements:

    - meeting_info: meeting-level metadata from meeting-info.json.
    - summary: business-facing extracted summary data from summary.json.
    - transcript_lines: sentence-level transcript data from transcript.json.

    Files such as speakers.json, events.json, and speaker-meta.json are counted
    for transparency but are not included here because their useful information
    is either duplicated in transcript_lines or unrelated to the assignment.
    """

    meeting_id: str
    meeting_info: dict[str, Any]
    summary: dict[str, Any]
    transcript_lines: list[dict[str, Any]]


@dataclass
class Taxonomy:
    """
    Keyword maps used by deterministic theme and call-type inference.

    theme_keywords maps high-level business themes to lowercase keyword hints.
    call_type_hints maps inferred call types to lowercase keyword hints.

    The values may come from the built-in fallback constants or from an external
    taxonomy.json file. In both cases, the downstream classifier treats them the
    same way: simple substring evidence produces repeatable scores.
    """

    theme_keywords: dict[str, list[str]]
    call_type_hints: dict[str, list[str]]


def load_dotenv() -> None:
    """
    Load local Postgres connection settings from .env files.

    The script supports two locations:

    - repo root .env, useful if the workspace already has shared database vars.
    - basics/iprep/meeting-analytics/.env, useful for assignment-specific credentials.

    Existing environment variables always win. This lets a shell session or CI
    job override local files without editing them. The parser is intentionally
    tiny because the file only needs simple KEY=value pairs.
    """

    for env_file in (SCRIPT_DIR.parent / ".env",):
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
    """
    Build a libpq-style Postgres DSN from environment variables.

    asyncpg accepts connection strings like postgresql://user:pass@host/db.
    Passwordless local development is also supported by omitting the password
    segment when PG_PASSWORD is empty.
    """

    if PASSWORD:
        return f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
    return f"postgresql://{USER}@{HOST}:{PORT}/{DATABASE}"


def quote_ident(name: str) -> str:
    """
    Quote a Postgres identifier such as a schema or table name.

    Values like schema names cannot be passed as bind parameters, so they must be
    inserted into SQL text. This helper escapes embedded double quotes and wraps
    the identifier in quotes to avoid accidental syntax issues.
    """

    return '"' + name.replace('"', '""') + '"'


def clean_topic(topic: str) -> str:
    """
    Normalize free-text topic or keyword strings for matching and deduping.

    The source topic tags are already fairly clean, but normalization makes the
    pipeline more robust by lowercasing, trimming edge whitespace, and collapsing
    repeated whitespace. The output is used for primary keys in summary_topics
    and for substring keyword matching.
    """

    return re.sub(r"\s+", " ", topic.strip().lower())


def normalize_keywords(values: list[Any]) -> list[str]:
    """
    Normalize keyword lists loaded from taxonomy config.

    Hand-edited JSON may include duplicate keywords, mixed casing, or values
    that are not strictly strings. This function converts everything to text,
    applies clean_topic(), removes empty values and duplicates, then sorts the
    result so taxonomy behavior is stable across runs.
    """

    return sorted({clean_topic(str(value)) for value in values if str(value).strip()})


def load_taxonomy(path: Path | None) -> Taxonomy:
    """
    Load taxonomy keyword maps from JSON, or return the built-in fallback.

    External taxonomy JSON is optional. When no file exists, the script uses the
    hand-reviewed THEME_KEYWORDS and CALL_TYPE_HINTS maps. When a file exists,
    it must contain:

    - themes[] items with theme_name and keyword_hints
    - call_types[] items with call_type and keyword_hints

    The function validates that both maps contain at least one usable entry.
    This prevents silently loading an empty taxonomy and producing no themes or
    call types.
    """

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
    """
    Convert JSON timestamp values into datetime objects for asyncpg.

    meeting-info.json stores timestamps as ISO strings ending in "Z". asyncpg
    expects Python datetime instances for timestamptz parameters, so this helper
    converts the UTC suffix into a Python-compatible +00:00 offset. Missing
    values stay NULL in Postgres.
    """

    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return None


def score_keywords(text: str, keyword_map: dict[str, list[str]]) -> Counter[str]:
    """
    Score labels by counting keyword matches in a piece of text.

    The keyword_map shape is {"label": ["keyword", ...]}. For each label, the
    score increases by one for every keyword that appears as a substring in the
    normalized text.

    This simple scoring is deliberate: it is transparent enough to explain in
    the take-home and easy to inspect when a meeting receives a surprising
    theme or call type.
    """

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
    """
    Infer business themes for a meeting from topics and key-moment signals.

    Inputs:
    - topics: summary.json topics, already extracted by the dataset provider.
    - key_moment_types: signal labels such as churn_signal, technical_issue,
      feature_gap, and concern.
    - summary: natural-language meeting summary, used as extra evidence when
      the meeting contains a concern.
    - taxonomy: theme keyword map from taxonomy.json or built-in fallback rules.

    Scoring logic:
    1. Each topic is matched against taxonomy.theme_keywords.
    2. Structured key-moment types add deterministic business signal boosts:
       - technical_issue supports Reliability and Support.
       - churn_signal strongly supports Commercial Risk.
       - feature_gap strongly supports Product Feedback / Roadmap.
    3. concern is intentionally broad, so it does not map to one theme by
       itself. Instead, the summary text is scanned for theme keywords.

    Return value:
    A Counter where keys are theme names and values are evidence counts. The
    caller stores every nonzero theme and marks the highest-scoring one as the
    primary theme.
    """

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
    """
    Infer the meeting's functional call type.

    Themes answer "what is this meeting about?" Call type answers "what kind of
    conversation is this?" The two are related but not identical. For example,
    an outage can appear in a customer escalation, an internal incident review,
    or a renewal-risk conversation.

    Scoring logic:
    1. Search combined topics + summary text against taxonomy.call_type_hints.
    2. Add small hints from theme_scores when themes imply likely intent:
       - Commercial Risk nudges sales_or_renewal.
       - Support / Escalation nudges support_escalation.
       - Internal Engineering nudges internal_planning.
       - Reliability without customer language nudges internal_incident.

    Confidence is a lightweight heuristic based on the winning score. It is not
    a calibrated probability; it simply helps identify meetings where the label
    was weakly supported and may deserve review.
    """

    scores: Counter[str] = Counter()
    combined = " ".join(topics + [summary])
    scores.update(score_keywords(combined, taxonomy.call_type_hints))

    if theme_scores["Customer Retention / Renewal / Commercial Risk"] > 0:
        scores["sales_or_renewal"] += 1
    if theme_scores["Support / Customer Escalation"] > 0:
        scores["support_escalation"] += 1
    if theme_scores["Internal Engineering / Planning / Execution"] > 0:
        scores["internal_planning"] += 1
    if theme_scores[
        "Reliability / Incidents / Outages"
    ] > 0 and "customer" not in clean_topic(combined):
        scores["internal_incident"] += 1

    if not scores:
        return "unknown", 0.0

    call_type, score = scores.most_common(1)[0]
    confidence = min(0.95, 0.45 + (score * 0.15))
    return call_type, confidence


def load_records(dataset_dir: Path) -> tuple[list[MeetingRecord], Counter[str]]:
    """
    Read meeting folders from disk and keep only requirement-relevant JSON.

    The function walks one directory per meeting under dataset_dir. It counts
    every JSON file type for the dry-run schema plan, but only loads files in
    SUPPORTED_FILES:

    - meeting-info.json for meeting metadata
    - summary.json for summaries, topics, action items, and key moments
    - transcript.json for sentence-level transcript and sentiment lines

    Returning file_counts alongside records makes the script transparent about
    what it saw and what it intentionally discarded.
    """

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
    """
    Create the Postgres schema and analytical tables.

    The tables are designed around the assignment's questions rather than the
    original JSON file boundaries:

    - meetings and meeting_participants describe meeting context.
    - meeting_summaries, summary_topics, action_items, and key_moments preserve
      the provider's extracted analytical fields.
    - transcript_lines keeps sentence-level sentiment evidence.
    - meeting_themes and call_types store deterministic taxonomy outputs.
    - sentiment_features stores derived metrics used for trend analysis.

    If reset is true, the target schema is dropped first. Otherwise tables are
    created if missing and then later truncated by truncate_tables().
    """

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
            action_item_count int NOT NULL,
            positive_pivot_count int NOT NULL,
            pricing_offer_count int NOT NULL
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
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_meeting_idx ON {q_schema}.{table}(meeting_id)"
        )


async def truncate_tables(conn: asyncpg.Connection, schema: str) -> None:
    """
    Empty all generated tables before reloading records.

    The loader is intended to be rerunnable while iterating on taxonomy rules.
    TRUNCATE is ordered from derived/child tables back to parent tables and uses
    CASCADE to satisfy foreign-key dependencies. RESTART IDENTITY is included
    for completeness even though the current schema uses natural keys.
    """

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
    """
    Split an action item into owner and action text when possible.

    summary.json action items are strings that often begin with
    "Person Name: action". This helper extracts that owner into a separate
    column for easier filtering. If no colon exists, or the prefix is too long
    to plausibly be a person/team name, the whole string remains the action
    text and owner is NULL.
    """

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
    """
    Insert meeting records and derived taxonomy/sentiment rows into Postgres.

    For each MeetingRecord, this function:

    1. Inserts meeting metadata and participants.
    2. Inserts summary, normalized topics, action items, and key moments.
    3. Inserts transcript lines with sentence-level sentiment.
    4. Runs infer_themes() to populate meeting_themes.
    5. Runs infer_call_type() to populate call_types.
    6. Aggregates transcript sentiment and key-moment counts into
       sentiment_features.

    This is the point where taxonomy generation becomes concrete database
    evidence. Every theme and call-type row is derived from explicit topics,
    key-moment types, summary text, and keyword rules.
    """

    q_schema = quote_ident(schema)

    for record in records:
        info = record.meeting_info
        summary = record.summary
        meeting_id = (
            summary.get("meetingId") or info.get("meetingId") or record.meeting_id
        )

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

        for role, emails in (
            ("all_email", info.get("allEmails", [])),
            ("invitee", info.get("invitees", [])),
        ):
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

        theme_scores = infer_themes(
            topics, key_moment_types, summary.get("summary") or "", taxonomy
        )
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
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
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
            key_moment_counts["positive_pivot"],
            key_moment_counts["pricing_offer"],
        )


def print_schema_plan(file_counts: Counter[str], records: list[MeetingRecord]) -> None:
    """
    Print a dry-run explanation of what the loader will do.

    This function exists for auditability. It shows:

    - Which source file types were detected.
    - Whether each file type is loaded or discarded.
    - Which analytical tables will be generated.
    - The most common observed topics.
    - The key-moment signal vocabulary.

    The output is useful when explaining why the schema does not mirror every
    JSON file and how the taxonomy was grounded in observed dataset fields.
    """

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
        topic_counts.update(
            clean_topic(str(t)) for t in record.summary.get("topics", [])
        )
        moment_counts.update(
            m.get("type") for m in record.summary.get("keyMoments", []) if m.get("type")
        )

    print("\nTop detected topics:")
    for topic, count in topic_counts.most_common(12):
        print(f"  {topic:35s} {count:4d}")

    print("\nDetected key moment signals:")
    for moment_type, count in moment_counts.most_common():
        print(f"  {moment_type:20s} {count:4d}")


async def print_counts(conn: asyncpg.Connection, schema: str) -> None:
    """
    Print row counts and starter SQL queries after a successful load.

    The row counts provide a quick sanity check that all meetings, topics,
    transcript lines, derived themes, call types, and sentiment features were
    inserted. The sample queries are meant for DBeaver exploration and map
    directly to the assignment's analysis questions.
    """

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
    print(
        f"  SELECT * FROM {schema}.meeting_themes WHERE is_primary ORDER BY theme, evidence_count DESC;"
    )
    print(
        f"  SELECT call_type, count(*), avg(negative_ratio) FROM {schema}.call_types JOIN {schema}.sentiment_features USING (meeting_id) GROUP BY 1;"
    )
    print(
        f"  SELECT theme, count(*) FROM {schema}.meeting_themes GROUP BY 1 ORDER BY 2 DESC;"
    )


async def main() -> None:
    """
    Command-line entry point for taxonomy generation and loading.

    Supported modes:

    - --dry-run:
      Inspect the dataset, print source-file decisions, and show taxonomy
      source without connecting to Postgres.

    - --reset:
      Drop and recreate the target schema before loading.

    - --taxonomy:
      Load taxonomy keyword hints from a JSON file. If the file is missing, the
      built-in fallback taxonomy is used.

    The function delays importing asyncpg until after --dry-run handling so that
    dataset/taxonomy inspection can run even in a Python environment that does
    not have the database driver installed.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY_PATH,
        help="Human-reviewed taxonomy JSON. Falls back to built-in rules if missing.",
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show inferred schema plan without connecting to Postgres.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset folder not found: {dataset_dir}")

    records, file_counts = load_records(dataset_dir)
    taxonomy = load_taxonomy(args.taxonomy)
    print_schema_plan(file_counts, records)
    print(
        f"\nTaxonomy source: {args.taxonomy if args.taxonomy.exists() else 'built-in fallback'}"
    )
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
