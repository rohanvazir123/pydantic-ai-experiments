"""
Take C — LLM-Assisted Semantic Clustering for Transcript Intelligence.

Pipeline:
  Extract topics → Dedup (exact + fuzzy) → Embed (nomic-embed-text via Ollama)
  → UMAP (10-dim) → HDBSCAN → Noise reassignment → LLM label clusters
  → Assign meetings to themes → Infer call types → Write outputs
  → Persist to Postgres (pgvector + tsvector via take_c_pg_store.IprepPhraseStore)

Why this approach over Take A/B:
  Topics are short semantic phrases; embeddings capture similarity that
  TF-IDF misses. HDBSCAN discovers cluster count from density — no manual K
  required. LLM labels run once per cluster (~10 calls), not per meeting.

See take_c_design.md for full tradeoff analysis.

Usage:
    python basics/iprep/i1/take_c_semantic_clustering.py
    python basics/iprep/i1/take_c_semantic_clustering.py --dry-run
    python basics/iprep/i1/take_c_semantic_clustering.py --min-cluster-size 4 --min-samples 2

Outputs (in basics/iprep/i1/cluster_work_c/):
    semantic_clusters.json  -- cluster definitions with LLM-generated labels
    meeting_themes.csv      -- per-meeting theme assignment + inferred call type
    phrase_clusters.csv     -- every topic phrase with its cluster assignment
    viz_coords.csv          -- 2D UMAP coordinates for scatter plot
    cluster_metrics.json    -- run metadata (params, noise ratio, timing)

  Postgres (iprep_i1_functional schema):
    semantic_clusters        -- cluster labels
    semantic_phrases         -- phrase embeddings (pgvector) + tsvector index
    semantic_meeting_themes  -- meeting → theme assignments + call type + sentiment

Environment variables (reads .env in repo root or script dir):
    EMBEDDING_BASE_URL, EMBEDDING_MODEL, EMBEDDING_API_KEY
    LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
    DATABASE_URL  (or PG_HOST/PORT/USER/PASSWORD/DATABASE for Take A compat)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from take_c_pg_store import IprepPhraseStore


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "cluster_work_c"

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
DEFAULT_LLM_MODEL = "llama3.1:8b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"

EMBEDDING_BATCH_SIZE = 50  # Ollama handles 50 phrases per batch comfortably

# UMAP params (design doc §3.4)
UMAP_N_COMPONENTS = 10
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0
UMAP_RANDOM_STATE = 42
UMAP_METRIC = "cosine"

# Concurrent LLM calls (avoid overwhelming Ollama)
LLM_CONCURRENCY = 5

# Phrases passed to the LLM per cluster label call
LABEL_SAMPLE_SIZE = 20

# Fuzzy dedup threshold
FUZZY_THRESHOLD = 90


# ---------------------------------------------------------------------------
# Environment loading (same pattern as generate_rule_based_taxonomy.py)
# ---------------------------------------------------------------------------


def _load_dotenv() -> None:
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


_load_dotenv()

EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", DEFAULT_BASE_URL)
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", DEFAULT_API_KEY)
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
LLM_MODEL: str = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
LLM_API_KEY: str = os.getenv("LLM_API_KEY", DEFAULT_API_KEY)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class MeetingRecord:
    meeting_id: str
    summary: dict[str, Any]


class TopicPhrase(BaseModel):
    canonical: str
    aliases: list[str] = Field(default_factory=list)
    cluster_id: int = -1
    embedding: list[float] = Field(default_factory=list)


class ClusterLabel(BaseModel):
    cluster_id: int
    theme_title: str
    audience: str  # Engineering | Product | Sales | All
    rationale: str
    representative_phrases: list[str] = Field(default_factory=list)


class MeetingThemeAssignment(BaseModel):
    meeting_id: str
    inferred_call_type: str
    call_confidence: str
    theme_ids: list[int]
    primary_theme_id: int
    sentiment_score: float | None
    overall_sentiment: str | None


# ---------------------------------------------------------------------------
# Step 1 — Load meeting records (raw JSON, no Postgres dependency)
# ---------------------------------------------------------------------------


def load_records(dataset_dir: Path) -> list[MeetingRecord]:
    records: list[MeetingRecord] = []
    for meeting_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        summary_path = meeting_dir / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        meeting_id = str(payload.get("meetingId") or meeting_dir.name)
        records.append(MeetingRecord(meeting_id=meeting_id, summary=payload))
    return records


# ---------------------------------------------------------------------------
# Step 2 — Extract and deduplicate topic phrases
# ---------------------------------------------------------------------------


def _clean_topic(topic: str) -> str:
    return re.sub(r"\s+", " ", topic.strip().lower())


def extract_topic_phrases(records: list[MeetingRecord]) -> list[TopicPhrase]:
    """Flatten all topics across meetings, exact-dedup, then fuzzy-dedup."""
    try:
        from rapidfuzz import fuzz
    except ImportError as exc:
        raise SystemExit("Install rapidfuzz first: pip install rapidfuzz") from exc

    # Collect all topics, normalized
    raw_topics: list[str] = []
    for record in records:
        for topic in record.summary.get("topics", []):
            raw_topics.append(_clean_topic(str(topic)))

    # Exact dedup preserving first-occurrence order
    seen: set[str] = set()
    unique: list[str] = []
    for t in raw_topics:
        if t and t not in seen:
            seen.add(t)
            unique.append(t)

    # Fuzzy dedup: group near-duplicates (token_sort_ratio >= FUZZY_THRESHOLD)
    # Keeps canonical (first seen in each group) → aliases mapping for traceability
    canonical_to_aliases: dict[str, list[str]] = {}
    for phrase in unique:
        matched: str | None = None
        for canonical in canonical_to_aliases:
            if fuzz.token_sort_ratio(phrase, canonical) >= FUZZY_THRESHOLD:
                matched = canonical
                break
        if matched:
            canonical_to_aliases[matched].append(phrase)
        else:
            canonical_to_aliases[phrase] = []

    phrases = [
        TopicPhrase(canonical=canonical, aliases=aliases)
        for canonical, aliases in canonical_to_aliases.items()
    ]

    print(f"  Raw topics collected:  {len(raw_topics)}")
    print(f"  After exact dedup:     {len(unique)}")
    print(f"  After fuzzy dedup:     {len(phrases)}")
    return phrases


# ---------------------------------------------------------------------------
# Step 3 — Generate embeddings
# ---------------------------------------------------------------------------


async def embed_phrases(phrases: list[TopicPhrase]) -> list[TopicPhrase]:
    """Batch-embed all canonical phrases using Ollama (nomic-embed-text)."""
    try:
        import openai
    except ImportError as exc:
        raise SystemExit("Install openai first: pip install openai") from exc

    client = openai.AsyncOpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    texts = [p.canonical for p in phrases]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        response = await client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend(data.embedding for data in response.data)
        print(f"  Embedded {min(i + EMBEDDING_BATCH_SIZE, len(texts))}/{len(texts)} phrases")

    for phrase, embedding in zip(phrases, all_embeddings):
        phrase.embedding = embedding

    await client.close()
    return phrases


# ---------------------------------------------------------------------------
# Step 4 — Dimensionality reduction
# ---------------------------------------------------------------------------


def _make_umap_reducer(n_components: int, min_dist: float) -> Any:
    try:
        import umap
    except ImportError as exc:
        raise SystemExit("Install umap-learn first: pip install umap-learn") from exc
    return umap.UMAP(
        n_components=n_components,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=min_dist,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )


def reduce_dimensions(embeddings: np.ndarray) -> np.ndarray:
    """UMAP to UMAP_N_COMPONENTS dims — used as HDBSCAN input."""
    reducer = _make_umap_reducer(n_components=UMAP_N_COMPONENTS, min_dist=UMAP_MIN_DIST)
    return reducer.fit_transform(embeddings)


def reduce_dimensions_2d(embeddings: np.ndarray) -> np.ndarray:
    """Separate 2D UMAP for visualization only — not used for clustering."""
    reducer = _make_umap_reducer(n_components=2, min_dist=0.1)
    return reducer.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Step 5 — HDBSCAN clustering + noise reassignment
# ---------------------------------------------------------------------------


def _compute_centroids(reduced: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    cluster_ids = sorted(set(labels) - {-1})
    return {cid: reduced[labels == cid].mean(axis=0) for cid in cluster_ids}


def _reassign_noise(reduced: np.ndarray, raw_labels: np.ndarray) -> np.ndarray:
    """Assign noise points (label=-1) to the nearest cluster centroid."""
    labels = raw_labels.copy()
    n_clusters = len(set(raw_labels) - {-1})
    if n_clusters == 0:
        labels[:] = 0
        return labels

    centroids = _compute_centroids(reduced, raw_labels)
    centroid_matrix = np.stack([centroids[cid] for cid in sorted(centroids)])
    cluster_ids_sorted = sorted(centroids)

    noise_indices = np.where(raw_labels == -1)[0]
    for idx in noise_indices:
        dists = np.linalg.norm(centroid_matrix - reduced[idx], axis=1)
        labels[idx] = cluster_ids_sorted[int(np.argmin(dists))]

    return labels


def cluster_phrases(
    phrases: list[TopicPhrase],
    reduced: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> tuple[list[TopicPhrase], dict[str, Any]]:
    """Run HDBSCAN, then reassign noise points to nearest centroid."""
    try:
        import hdbscan
    except ImportError as exc:
        raise SystemExit("Install hdbscan first: pip install hdbscan") from exc

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",  # euclidean in UMAP-reduced space
        cluster_selection_method="eom",
    )
    raw_labels: np.ndarray = clusterer.fit_predict(reduced)

    n_clusters = len(set(raw_labels) - {-1})
    n_noise = int(np.sum(raw_labels == -1))
    noise_pct = n_noise / len(phrases) * 100
    print(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise points ({noise_pct:.1f}%)")

    final_labels = _reassign_noise(reduced, raw_labels)

    for phrase, label in zip(phrases, final_labels):
        phrase.cluster_id = int(label)

    metrics: dict[str, Any] = {
        "phrase_count": len(phrases),
        "n_clusters": n_clusters,
        "n_noise_raw": n_noise,
        "noise_ratio": round(n_noise / len(phrases), 3),
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "umap_n_components": UMAP_N_COMPONENTS,
        "umap_n_neighbors": UMAP_N_NEIGHBORS,
        "umap_metric": UMAP_METRIC,
        "umap_random_state": UMAP_RANDOM_STATE,
    }
    return phrases, metrics


# ---------------------------------------------------------------------------
# Step 6 — LLM cluster labeling
# ---------------------------------------------------------------------------


_LABEL_PROMPT = """\
You are categorizing customer call topics for a B2B SaaS company's leadership team.

The following phrases all come from one theme cluster discovered by semantic clustering.
Based only on these phrases, provide a short, executive-level theme label.

Phrases:
{phrases}

Respond with valid JSON only — no extra text, no markdown fences:
{{"theme_title": "<3-6 words, title case>", "audience": "<Engineering | Product | Sales | All>", "rationale": "<one sentence: why this theme matters to that audience>"}}"""


def _extract_json(content: str) -> dict[str, str] | None:
    """Pull the first JSON object from an LLM response, handling markdown fences."""
    text = re.sub(r"```(?:json)?", "", content).replace("```", "").strip()
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


async def label_clusters(phrases: list[TopicPhrase]) -> list[ClusterLabel]:
    """Call the LLM once per cluster (with semaphore) to generate leadership labels."""
    try:
        import openai
    except ImportError as exc:
        raise SystemExit("Install openai first: pip install openai") from exc

    client = openai.AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

    # Group phrases by cluster_id
    by_cluster: dict[int, list[str]] = {}
    for phrase in phrases:
        by_cluster.setdefault(phrase.cluster_id, []).append(phrase.canonical)

    async def label_one(cluster_id: int, cluster_phrases: list[str]) -> ClusterLabel:
        sample = cluster_phrases[:LABEL_SAMPLE_SIZE]
        prompt = _LABEL_PROMPT.format(phrases="\n".join(f"- {p}" for p in sample))

        async with semaphore:
            for attempt in range(3):
                try:
                    response = await client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    parsed = _extract_json(response.choices[0].message.content or "")
                    if parsed and "theme_title" in parsed:
                        return ClusterLabel(
                            cluster_id=cluster_id,
                            theme_title=parsed.get("theme_title", f"Cluster {cluster_id}"),
                            audience=parsed.get("audience", "All"),
                            rationale=parsed.get("rationale", ""),
                            representative_phrases=sample[:10],
                        )
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        print(f"  Warning: label failed for cluster {cluster_id}: {exc}")

        # Fallback: derive title from top phrases
        return ClusterLabel(
            cluster_id=cluster_id,
            theme_title=" / ".join(sample[:3]) if sample else f"Cluster {cluster_id}",
            audience="All",
            rationale="LLM labeling failed — manual review needed.",
            representative_phrases=sample[:10],
        )

    labels = await asyncio.gather(*[label_one(cid, cp) for cid, cp in sorted(by_cluster.items())])
    await client.close()
    return list(labels)


# ---------------------------------------------------------------------------
# Step 7 — Assign meetings to themes
# ---------------------------------------------------------------------------


def assign_meetings_to_themes(
    records: list[MeetingRecord],
    phrases: list[TopicPhrase],
    labels: list[ClusterLabel],
    call_types: dict[str, tuple[str, str]],
) -> list[MeetingThemeAssignment]:
    # canonical + all aliases → cluster_id lookup
    phrase_to_cluster: dict[str, int] = {}
    for phrase in phrases:
        phrase_to_cluster[phrase.canonical] = phrase.cluster_id
        for alias in phrase.aliases:
            phrase_to_cluster[alias] = phrase.cluster_id

    label_by_id: dict[int, ClusterLabel] = {lb.cluster_id: lb for lb in labels}

    assignments: list[MeetingThemeAssignment] = []
    for record in records:
        topics = [_clean_topic(str(t)) for t in record.summary.get("topics", [])]
        cluster_hits: Counter[int] = Counter()
        for topic in topics:
            cid = phrase_to_cluster.get(topic)
            if cid is not None and cid in label_by_id:
                cluster_hits[cid] += 1

        theme_ids = sorted(cluster_hits.keys())
        primary_theme_id = cluster_hits.most_common(1)[0][0] if cluster_hits else -1
        call_type, confidence = call_types.get(record.meeting_id, ("unknown", "low"))

        assignments.append(
            MeetingThemeAssignment(
                meeting_id=record.meeting_id,
                inferred_call_type=call_type,
                call_confidence=confidence,
                theme_ids=theme_ids,
                primary_theme_id=primary_theme_id,
                sentiment_score=record.summary.get("sentimentScore"),
                overall_sentiment=record.summary.get("overallSentiment"),
            )
        )

    return assignments


# ---------------------------------------------------------------------------
# Step 8 — Infer call types via direct LLM classification
# ---------------------------------------------------------------------------


_CALL_TYPE_PROMPT = """\
Classify this B2B SaaS meeting as exactly one of three call types.

Call types:
- support: Customer contacting the vendor about an issue, bug, escalation, outage, or billing problem. Reactive, customer-initiated.
- external: Account manager or sales rep meeting with a customer for renewal, QBR, roadmap review, product demo, expansion discussion, or feedback session.
- internal: Internal team meeting — engineering sync, sprint planning, incident post-mortem, architecture review, cross-team coordination, or capacity planning.

Meeting summary:
{summary}

Respond with valid JSON only — no extra text, no markdown fences:
{{"call_type": "<support | external | internal>", "confidence": "<high | medium | low>"}}"""


async def infer_call_types(records: list[MeetingRecord]) -> dict[str, tuple[str, str]]:
    """Classify every meeting's call type in parallel, with concurrency limit."""
    try:
        import openai
    except ImportError as exc:
        raise SystemExit("Install openai first: pip install openai") from exc

    client = openai.AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

    async def classify_one(record: MeetingRecord) -> tuple[str, tuple[str, str]]:
        summary_text = record.summary.get("summary", "")
        prompt = _CALL_TYPE_PROMPT.format(summary=summary_text[:1500])

        async with semaphore:
            for attempt in range(3):
                try:
                    response = await client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                    )
                    parsed = _extract_json(response.choices[0].message.content or "")
                    if parsed and "call_type" in parsed:
                        call_type = parsed["call_type"]
                        confidence = parsed.get("confidence", "medium")
                        return record.meeting_id, (call_type, confidence)
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        print(f"  Warning: call-type inference failed for {record.meeting_id}: {exc}")

        return record.meeting_id, ("unknown", "low")

    results = await asyncio.gather(*[classify_one(r) for r in records])
    await client.close()
    return dict(results)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_outputs(
    output_dir: Path,
    phrases: list[TopicPhrase],
    labels: list[ClusterLabel],
    assignments: list[MeetingThemeAssignment],
    metrics: dict[str, Any],
    viz_coords: np.ndarray | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_by_id: dict[int, ClusterLabel] = {lb.cluster_id: lb for lb in labels}

    # semantic_clusters.json
    (output_dir / "semantic_clusters.json").write_text(
        json.dumps(
            [
                {
                    "cluster_id": lb.cluster_id,
                    "theme_title": lb.theme_title,
                    "audience": lb.audience,
                    "rationale": lb.rationale,
                    "representative_phrases": lb.representative_phrases,
                    "phrase_count": sum(1 for p in phrases if p.cluster_id == lb.cluster_id),
                }
                for lb in sorted(labels, key=lambda x: x.cluster_id)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    # phrase_clusters.csv
    with (output_dir / "phrase_clusters.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["cluster_id", "theme_title", "canonical", "aliases"])
        writer.writeheader()
        for phrase in sorted(phrases, key=lambda p: (p.cluster_id, p.canonical)):
            lb = label_by_id.get(phrase.cluster_id)
            writer.writerow(
                {
                    "cluster_id": phrase.cluster_id,
                    "theme_title": lb.theme_title if lb else "",
                    "canonical": phrase.canonical,
                    "aliases": "; ".join(phrase.aliases),
                }
            )

    # meeting_themes.csv
    with (output_dir / "meeting_themes.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "meeting_id",
                "primary_theme_id",
                "primary_theme_title",
                "primary_theme_audience",
                "all_theme_ids",
                "all_theme_titles",
                "call_type",
                "call_confidence",
                "sentiment_score",
                "overall_sentiment",
            ],
        )
        writer.writeheader()
        for a in assignments:
            primary_lb = label_by_id.get(a.primary_theme_id)
            all_titles = "; ".join(
                label_by_id[tid].theme_title for tid in a.theme_ids if tid in label_by_id
            )
            writer.writerow(
                {
                    "meeting_id": a.meeting_id,
                    "primary_theme_id": a.primary_theme_id,
                    "primary_theme_title": primary_lb.theme_title if primary_lb else "",
                    "primary_theme_audience": primary_lb.audience if primary_lb else "",
                    "all_theme_ids": "; ".join(str(t) for t in a.theme_ids),
                    "all_theme_titles": all_titles,
                    "call_type": a.inferred_call_type,
                    "call_confidence": a.call_confidence,
                    "sentiment_score": a.sentiment_score,
                    "overall_sentiment": a.overall_sentiment,
                }
            )

    # cluster_metrics.json
    (output_dir / "cluster_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # viz_coords.csv (for scatter plot, optional)
    if viz_coords is not None:
        with (output_dir / "viz_coords.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["canonical", "cluster_id", "theme_title", "x", "y"]
            )
            writer.writeheader()
            for phrase, (x, y) in zip(phrases, viz_coords):
                lb = label_by_id.get(phrase.cluster_id)
                writer.writerow(
                    {
                        "canonical": phrase.canonical,
                        "cluster_id": phrase.cluster_id,
                        "theme_title": lb.theme_title if lb else "",
                        "x": round(float(x), 4),
                        "y": round(float(y), 4),
                    }
                )

    print(f"\nOutputs written to: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")


def print_summary(labels: list[ClusterLabel], assignments: list[MeetingThemeAssignment]) -> None:
    label_by_id: dict[int, ClusterLabel] = {lb.cluster_id: lb for lb in labels}
    theme_counts: Counter[int] = Counter(a.primary_theme_id for a in assignments)
    call_type_counts: Counter[str] = Counter(a.inferred_call_type for a in assignments)
    sentiment_by_theme: dict[int, list[float]] = {}
    for a in assignments:
        if a.sentiment_score is not None:
            sentiment_by_theme.setdefault(a.primary_theme_id, []).append(a.sentiment_score)

    print("\n─── Discovered Themes ───────────────────────────────────────")
    for lb in sorted(labels, key=lambda x: -theme_counts.get(x.cluster_id, 0)):
        count = theme_counts.get(lb.cluster_id, 0)
        scores = sentiment_by_theme.get(lb.cluster_id, [])
        avg_sentiment = f"{sum(scores)/len(scores):.2f}" if scores else "n/a"
        print(f"\n  [{lb.cluster_id}] {lb.theme_title}  ({lb.audience})")
        print(f"       {lb.rationale}")
        print(f"       {count} meetings  |  avg sentiment: {avg_sentiment}")
        print(f"       sample phrases: {', '.join(lb.representative_phrases[:4])}")

    print("\n─── Call Type Distribution ──────────────────────────────────")
    for call_type, count in sorted(call_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {call_type:12s}  {count:3d} meetings")


# ---------------------------------------------------------------------------
# Postgres persistence
# ---------------------------------------------------------------------------


async def persist_to_postgres(
    store: IprepPhraseStore,
    phrases: list[TopicPhrase],
    labels: list[ClusterLabel],
    assignments: list[MeetingThemeAssignment],
    reset: bool = False,
) -> None:
    """Write clustering results into the iprep_i1_functional Postgres schema."""
    if reset:
        print("  Resetting semantic tables...")
        await store.reset_semantic_tables()
    else:
        await store.initialize()

    label_by_id = {lb.cluster_id: lb for lb in labels}

    # Cluster labels
    cluster_dicts = [
        {
            "cluster_id": lb.cluster_id,
            "theme_title": lb.theme_title,
            "audience": lb.audience,
            "rationale": lb.rationale,
            "phrase_count": sum(1 for p in phrases if p.cluster_id == lb.cluster_id),
        }
        for lb in labels
    ]
    await store.save_cluster_labels(cluster_dicts)
    print(f"  Saved {len(cluster_dicts)} clusters")

    # Phrase embeddings
    phrase_dicts = [
        {
            "canonical": p.canonical,
            "aliases": p.aliases,
            "cluster_id": p.cluster_id,
            "embedding": p.embedding,
        }
        for p in phrases
    ]
    await store.save_phrases(phrase_dicts)
    print(f"  Saved {len(phrase_dicts)} phrase embeddings")

    # Meeting theme assignments
    assignment_dicts = [
        {
            "meeting_id": a.meeting_id,
            "theme_ids": a.theme_ids,
            "primary_theme_id": a.primary_theme_id,
            "inferred_call_type": a.inferred_call_type,
            "call_confidence": a.call_confidence,
            "sentiment_score": a.sentiment_score,
            "overall_sentiment": a.overall_sentiment,
        }
        for a in assignments
    ]
    await store.save_meeting_themes(assignment_dicts)
    print(f"  Saved {len(assignment_dicts)} meeting-theme assignments")

    counts = await store.row_counts()
    print(f"  Row counts: {counts}")


async def print_pg_insights(store: IprepPhraseStore) -> None:
    """Print the four canned insight queries against the persisted data."""
    print("\n─── Insight: Theme Sentiment ────────────────────────────────")
    rows = await store.insight_theme_sentiment()
    for r in rows:
        bar = "+" if (r["avg_sentiment"] or 0) >= 3.5 else "-"
        print(f"  [{bar}] {r['theme_title']:<40s}  "
              f"avg={r['avg_sentiment']}  meetings={r['meeting_count']}")

    print("\n─── Insight: Churn Risk by Theme ────────────────────────────")
    rows = await store.insight_churn_by_theme()
    if rows:
        for r in rows:
            print(f"  {r['theme_title']:<40s}  churn_signals={r['churn_signal_count']}  "
                  f"per_meeting={r['churn_per_meeting']}")
    else:
        print("  (key_moments table not found — run Take A first)")

    print("\n─── Insight: Call Type x Theme Matrix ───────────────────────")
    rows = await store.insight_call_type_theme_matrix()
    current_ct = None
    for r in rows:
        if r["call_type"] != current_ct:
            current_ct = r["call_type"]
            print(f"\n  [{current_ct}]")
        print(f"    {r['theme_title']:<40s}  {r['meeting_count']} meetings")

    print("\n─── Insight: Feature Gap Themes ─────────────────────────────")
    rows = await store.insight_feature_gap_themes()
    if rows:
        for r in rows[:8]:
            print(f"  {r['theme_title']:<40s}  feature_gaps={r['feature_gap_count']}")
    else:
        print("  (key_moments table not found — run Take A first)")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Take C: LLM-assisted semantic clustering of meeting topics."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-cluster-size", type=int, default=5,
        help="HDBSCAN min_cluster_size. Lower = more clusters. (default: 5)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=3,
        help="HDBSCAN min_samples. Higher = more noise points. (default: 3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load and deduplicate topics, then exit without calling Ollama.",
    )
    parser.add_argument(
        "--skip-viz", action="store_true",
        help="Skip 2D UMAP for visualization (saves ~10s).",
    )
    parser.add_argument(
        "--skip-pg", action="store_true",
        help="Skip Postgres persistence step.",
    )
    parser.add_argument(
        "--reset-pg", action="store_true",
        help="Drop and recreate semantic tables before inserting.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset folder not found: {dataset_dir}")

    t_start = time.monotonic()
    print("=== Take C: LLM-Assisted Semantic Clustering ===")
    print(f"  Embedding model : {EMBEDDING_MODEL}  ({EMBEDDING_BASE_URL})")
    print(f"  LLM model       : {LLM_MODEL}  ({LLM_BASE_URL})")
    print(f"  Postgres        : {'skip' if args.skip_pg else 'enabled (iprep_i1_functional)'}")

    # Step 1
    print("\n[1/9] Loading meeting records...")
    records = load_records(dataset_dir)
    print(f"  Loaded {len(records)} meetings")

    # Step 2
    print("\n[2/9] Extracting and deduplicating topic phrases...")
    phrases = extract_topic_phrases(records)

    if args.dry_run:
        print("\n[dry-run] Stopping before Ollama calls.")
        print("  Sample phrases (first 15):")
        for phrase in phrases[:15]:
            alias_str = f"  (also: {', '.join(phrase.aliases)})" if phrase.aliases else ""
            print(f"    {phrase.canonical}{alias_str}")
        return

    # Step 3
    print("\n[3/9] Generating embeddings...")
    phrases = await embed_phrases(phrases)

    # Step 4
    print("\n[4/9] Reducing dimensions with UMAP...")
    embeddings_matrix = np.array([p.embedding for p in phrases])
    reduced = reduce_dimensions(embeddings_matrix)

    viz_coords: np.ndarray | None = None
    if not args.skip_viz:
        print("  Computing 2D UMAP for visualization...")
        viz_coords = reduce_dimensions_2d(embeddings_matrix)

    # Step 5
    print("\n[5/9] Clustering with HDBSCAN...")
    phrases, metrics = cluster_phrases(
        phrases=phrases,
        reduced=reduced,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    # Step 6
    print("\n[6/9] Labeling clusters with LLM...")
    labels = await label_clusters(phrases)
    for lb in sorted(labels, key=lambda x: x.cluster_id):
        print(f"  [{lb.cluster_id}] {lb.theme_title}  ({lb.audience})")

    # Step 7 (call types run before theme assignment so we can pass them together)
    print("\n[7/9] Inferring call types...")
    call_types = await infer_call_types(records)
    call_type_dist = Counter(v for v, _ in call_types.values())
    print(f"  Classified {len(call_types)} meetings: {dict(call_type_dist)}")

    # Step 8
    print("\n[8/9] Assigning meetings to themes...")
    assignments = assign_meetings_to_themes(records, phrases, labels, call_types)

    elapsed = time.monotonic() - t_start
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["llm_model"] = LLM_MODEL
    metrics["embedding_model"] = EMBEDDING_MODEL
    metrics["meeting_count"] = len(records)

    output_dir = args.output_dir.resolve()
    write_outputs(output_dir, phrases, labels, assignments, metrics, viz_coords)
    print_summary(labels, assignments)

    # Step 9 — persist to Postgres (pgvector + tsvector)
    if not args.skip_pg:
        print("\n[9/9] Persisting to Postgres (iprep_i1_functional)...")
        store = IprepPhraseStore()
        try:
            await persist_to_postgres(
                store, phrases, labels, assignments, reset=args.reset_pg
            )
            await print_pg_insights(store)
        finally:
            await store.close()
    else:
        print("\n[9/9] Postgres step skipped (--skip-pg)")

    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
