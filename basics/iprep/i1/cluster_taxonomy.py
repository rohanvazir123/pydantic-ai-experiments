"""
Discover transcript themes with unsupervised TF-IDF clustering.

This script is the scalable alternative to hand-written THEME_KEYWORDS.
It does not need seed mappings. Instead, it:

1. Builds one analysis document per meeting from summary.json:
   summary + topics + key moment text/types + action items.
2. Converts those meeting documents into TF-IDF vectors.
3. Clusters the vectors with KMeans.
4. Names each cluster using the highest-scoring TF-IDF terms near that cluster.
5. Writes machine-readable outputs for review.

Why this is useful:
Rule-based mappings are easy to defend for a small take-home dataset, but they
do not scale well when products, customers, and topic vocabulary change. TF-IDF
clustering gives a data-driven discovery layer: it can reveal recurring themes
without requiring someone to maintain every synonym up front.

Important limitation:
The generated cluster labels are descriptive, not authoritative business
taxonomy names. Top TF-IDF terms tell you what words distinguish a cluster; a
human should still review examples and decide whether the cluster should become
a stable business category.

How to interpret quality:
This is unsupervised learning, so there is no ground-truth accuracy score unless
someone manually labels meetings. The script therefore reports silhouette score:

- Range: -1 to 1
- Higher is better
- Near 0 means clusters overlap heavily
- Negative means many points may fit another cluster better

For this dataset, a low silhouette score should not be treated as a failure of
the code. It means the meetings are semantically blended: renewals can involve
outages, compliance can involve onboarding, and incidents can create churn risk.
That is exactly why clustering should be presented as a discovery/validation
tool rather than the final production classifier.

Recommended interpretation:
- Use clustering to find candidate themes and taxonomy drift.
- Use top TF-IDF terms and example meetings to inspect each cluster.
- Promote stable, business-meaningful clusters into a reviewed taxonomy.
- Use rule-based or supervised classification for repeatable final reporting.

Usage:
    python basics/iprep/i1/cluster_taxonomy.py
    python basics/iprep/i1/cluster_taxonomy.py --clusters 8
    python basics/iprep/i1/cluster_taxonomy.py --auto-k --min-clusters 4 --max-clusters 12

Outputs:
    basics/iprep/i1/cluster_work/meeting_clusters.csv
    basics/iprep/i1/cluster_work/cluster_summary.json
    basics/iprep/i1/cluster_work/cluster_terms.csv
    basics/iprep/i1/cluster_work/cluster_metrics.json
    basics/iprep/i1/cluster_work/cluster_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "cluster_work"


@dataclass
class MeetingDocument:
    """
    Text and metadata used as one clustering unit.

    Each instance represents one meeting. The clustering algorithm only consumes
    document_text, but the remaining fields are preserved so output files can
    show interpretable examples: title, topics, sentiment, and summary.

    Keeping metadata alongside the vectorized text is important because cluster
    evaluation is partly qualitative. A silhouette score can tell us whether
    vectors are separated, but humans still need examples to decide whether a
    cluster is business-meaningful.
    """

    meeting_id: str
    title: str
    summary: str
    topics: list[str]
    action_items: list[str]
    key_moment_types: list[str]
    key_moment_texts: list[str]
    overall_sentiment: str | None
    sentiment_score: float | None
    document_text: str


def clean_text(value: str) -> str:
    """
    Normalize text enough for readable outputs and stable vectorization.

    TF-IDF handles lowercasing and tokenization later. This function only trims
    whitespace and collapses repeated spaces so generated documents and CSV
    outputs stay legible.
    """

    return re.sub(r"\s+", " ", value.strip())


def load_json(path: Path) -> Any:
    """
    Read a UTF-8 JSON file from disk.

    The dataset is small enough that loading complete JSON files is simpler and
    clearer than streaming. Returning Any keeps this helper generic for both
    summary.json and meeting-info.json payloads.
    """

    return json.loads(path.read_text(encoding="utf-8"))


def repeat_terms(values: list[str], repeat: int) -> str:
    """
    Repeat short structured tags so TF-IDF sees them as important.

    Summary prose can be much longer than topic tags. Repeating topics and key
    moment types gives those curated fields more influence without needing seed
    mappings.
    """

    return " ".join(clean_text(value) for value in values for _ in range(repeat) if value)


def build_document_text(
    summary: str,
    topics: list[str],
    action_items: list[str],
    key_moment_types: list[str],
    key_moment_texts: list[str],
) -> str:
    """
    Combine useful fields into the text representation used for clustering.

    The document intentionally excludes speaker join/leave events and raw
    speaker metadata because those fields do not describe business themes. It
    emphasizes topics and key-moment types because they are compact, structured
    signals already provided by the dataset.
    """

    parts = [
        summary,
        repeat_terms(topics, repeat=4),
        repeat_terms(key_moment_types, repeat=3),
        " ".join(key_moment_texts),
        " ".join(action_items),
    ]
    return clean_text(" ".join(part for part in parts if part))


def load_meeting_documents(dataset_dir: Path) -> list[MeetingDocument]:
    """
    Load one MeetingDocument per meeting directory.

    Only summary.json and meeting-info.json are needed. transcript.json can be
    useful for deeper semantic clustering, but summary.json already contains
    distilled content and avoids making clusters depend on conversational filler.
    """

    documents: list[MeetingDocument] = []
    for meeting_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        summary_path = meeting_dir / "summary.json"
        info_path = meeting_dir / "meeting-info.json"
        if not summary_path.exists():
            continue

        summary_payload = load_json(summary_path)
        info_payload = load_json(info_path) if info_path.exists() else {}
        key_moments = summary_payload.get("keyMoments", [])
        topics = [str(topic) for topic in summary_payload.get("topics", [])]
        action_items = [str(item) for item in summary_payload.get("actionItems", [])]
        key_moment_types = [
            str(moment.get("type"))
            for moment in key_moments
            if moment.get("type")
        ]
        key_moment_texts = [
            str(moment.get("text"))
            for moment in key_moments
            if moment.get("text")
        ]

        summary = str(summary_payload.get("summary") or "")
        document_text = build_document_text(
            summary=summary,
            topics=topics,
            action_items=action_items,
            key_moment_types=key_moment_types,
            key_moment_texts=key_moment_texts,
        )

        documents.append(
            MeetingDocument(
                meeting_id=str(summary_payload.get("meetingId") or meeting_dir.name),
                title=str(info_payload.get("title") or ""),
                summary=summary,
                topics=topics,
                action_items=action_items,
                key_moment_types=key_moment_types,
                key_moment_texts=key_moment_texts,
                overall_sentiment=summary_payload.get("overallSentiment"),
                sentiment_score=summary_payload.get("sentimentScore"),
                document_text=document_text,
            )
        )

    return documents


def require_sklearn() -> tuple[Any, Any, Any, Any]:
    """
    Import sklearn pieces lazily and provide a clear installation error.

    The rest of the project can still be inspected without scikit-learn. The
    import happens only when clustering actually runs, which keeps syntax
    checks and code review lightweight.
    """

    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import normalize
    except ImportError as exc:
        raise SystemExit(
            "Install scikit-learn first:\n"
            "  pip install scikit-learn\n\n"
            "This script uses sklearn for TF-IDF vectorization, KMeans clustering, "
            "and optional silhouette scoring."
        ) from exc

    return KMeans, TfidfVectorizer, silhouette_score, normalize


def vectorize_documents(documents: list[MeetingDocument], max_features: int):
    """
    Convert meeting documents to a TF-IDF matrix.

    N-grams up to length three are included so phrases like "churn risk",
    "incident response", and "compliance reporting" can surface as cluster
    terms. English stop words are removed to keep labels focused.
    """

    # L2 normalization makes cosine-like geometry work well for KMeans on text.
    _, TfidfVectorizer, _, normalize = require_sklearn()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.85,
        max_features=max_features,
    )
    matrix = vectorizer.fit_transform(document.document_text for document in documents)
    return vectorizer, normalize(matrix)


def compute_silhouette_score(matrix: Any, labels: list[int]) -> float | None:
    """
    Compute silhouette score for a fitted clustering result.

    Silhouette compares each point's distance to its own cluster against its
    distance to the nearest other cluster. The range is -1 to 1, where higher is
    better. It is undefined when there is only one cluster or every point is its
    own cluster, so this function returns None in those cases.
    """

    # Silhouette is undefined for a single cluster or one cluster per record.
    unique_labels = set(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return None

    _, _, silhouette_score, _ = require_sklearn()
    return float(silhouette_score(matrix, labels))


def score_cluster_counts(
    matrix: Any,
    min_clusters: int,
    max_clusters: int,
    random_state: int,
) -> list[dict[str, float | int]]:
    """
    Fit several k values and return their silhouette scores.

    This provides a simple quantitative way to compare possible cluster counts.
    It does not prove the clusters are semantically correct, but it helps avoid
    picking a k where clusters are obviously overlapping or poorly separated.
    """

    # Refit KMeans for each candidate k, then score the fitted labels.
    KMeans, _, _, _ = require_sklearn()
    max_allowed = min(max_clusters, matrix.shape[0] - 1)
    rows: list[dict[str, float | int]] = []

    for k in range(min_clusters, max_allowed + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = list(model.fit_predict(matrix))
        score = compute_silhouette_score(matrix, labels)
        rows.append({"clusters": k, "silhouette_score": score if score is not None else float("nan")})

    return rows


def choose_cluster_count(matrix: Any, min_clusters: int, max_clusters: int, random_state: int) -> tuple[int, list[dict[str, float | int]]]:
    """
    Pick k by silhouette score over a small candidate range.

    This is a convenience for exploration, not a universal truth. For a
    presentation, k should still be chosen for interpretability: enough clusters
    to separate meaningful themes, not so many that leadership gets a taxonomy
    with twenty tiny buckets.
    """

    # Pick the highest silhouette score, but still return every score so the
    # reviewer can see whether the winner is meaningfully better.
    score_rows = score_cluster_counts(matrix, min_clusters, max_clusters, random_state)
    best_k = int(score_rows[0]["clusters"])
    best_score = -1.0

    for row in score_rows:
        k = int(row["clusters"])
        score = float(row["silhouette_score"])
        if score > best_score:
            best_k = k
            best_score = score

    return best_k, score_rows


def fit_clusters(matrix: Any, clusters: int, random_state: int):
    """
    Fit KMeans and return the trained model plus meeting-level labels.

    KMeans is used because it is simple, fast, deterministic with random_state,
    and easy to explain. It does require choosing k, which is why the script
    supports both fixed --clusters and exploratory --auto-k modes.
    """

    KMeans, _, _, _ = require_sklearn()
    model = KMeans(n_clusters=clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(matrix)
    return model, labels


def top_terms_for_clusters(vectorizer: Any, model: Any, top_n: int) -> dict[int, list[str]]:
    """
    Extract the highest-weight TF-IDF terms for each cluster centroid.

    These terms are the unsupervised equivalent of seed keywords. They are not
    supplied by us. They are learned from the documents and help a human
    understand what distinguishes each cluster.
    """

    feature_names = vectorizer.get_feature_names_out()
    cluster_terms: dict[int, list[str]] = {}

    for cluster_id, centroid in enumerate(model.cluster_centers_):
        top_indices = centroid.argsort()[::-1][:top_n]
        cluster_terms[cluster_id] = [feature_names[index] for index in top_indices]

    return cluster_terms


def summarize_clusters(
    documents: list[MeetingDocument],
    labels: list[int],
    cluster_terms: dict[int, list[str]],
    examples_per_cluster: int,
) -> list[dict[str, Any]]:
    """
    Build a JSON-friendly summary for each cluster.

    The label is intentionally mechanical: it joins the top few TF-IDF terms.
    That makes it clear the name is generated from evidence, not manually mapped
    from a seed taxonomy. The examples help a reviewer decide whether to rename,
    merge, split, or promote the cluster.

    The summary includes three complementary views:

    - top_terms: vector-space terms that distinguish the cluster.
    - top_topics: original summary topic tags, useful because they are already
      curated signals from the source data.
    - key_moment_types: business signals like churn_signal or feature_gap that
      explain why the cluster may matter.
    """

    by_cluster: dict[int, list[MeetingDocument]] = {}
    for document, label in zip(documents, labels, strict=True):
        by_cluster.setdefault(int(label), []).append(document)

    summaries: list[dict[str, Any]] = []
    for cluster_id in sorted(by_cluster):
        cluster_docs = by_cluster[cluster_id]
        topic_counts: Counter[str] = Counter()
        moment_counts: Counter[str] = Counter()
        sentiment_counts: Counter[str] = Counter()

        for document in cluster_docs:
            topic_counts.update(document.topics)
            moment_counts.update(document.key_moment_types)
            if document.overall_sentiment:
                sentiment_counts[document.overall_sentiment] += 1

        top_terms = cluster_terms[cluster_id]
        summaries.append(
            {
                "cluster_id": cluster_id,
                "generated_label": " / ".join(top_terms[:4]),
                "meeting_count": len(cluster_docs),
                "top_terms": top_terms,
                "top_topics": topic_counts.most_common(10),
                "key_moment_types": moment_counts.most_common(),
                "overall_sentiments": sentiment_counts.most_common(),
                "examples": [
                    {
                        "meeting_id": document.meeting_id,
                        "title": document.title,
                        "topics": document.topics,
                        "summary": document.summary,
                    }
                    for document in cluster_docs[:examples_per_cluster]
                ],
            }
        )

    return summaries


def write_outputs(
    output_dir: Path,
    documents: list[MeetingDocument],
    labels: list[int],
    cluster_terms: dict[int, list[str]],
    cluster_summary: list[dict[str, Any]],
    metrics: dict[str, Any],
    score_rows: list[dict[str, float | int]],
) -> None:
    """
    Write CSV and JSON artifacts for review and downstream analysis.

    meeting_clusters.csv is the easiest file to inspect in spreadsheet tools.
    cluster_terms.csv shows the evidence behind generated labels.
    cluster_summary.json keeps richer examples and aggregate counts.
    cluster_metrics.json stores the selected run's silhouette score.
    cluster_scores.csv stores candidate k scores, especially useful with
    --auto-k.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "meeting_clusters.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "meeting_id",
                "cluster_id",
                "generated_cluster_label",
                "title",
                "overall_sentiment",
                "sentiment_score",
                "topics",
            ],
        )
        writer.writeheader()
        for document, label in zip(documents, labels, strict=True):
            terms = cluster_terms[int(label)]
            writer.writerow(
                {
                    "meeting_id": document.meeting_id,
                    "cluster_id": int(label),
                    "generated_cluster_label": " / ".join(terms[:4]),
                    "title": document.title,
                    "overall_sentiment": document.overall_sentiment,
                    "sentiment_score": document.sentiment_score,
                    "topics": "; ".join(document.topics),
                }
            )

    with (output_dir / "cluster_terms.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cluster_id", "rank", "term"])
        writer.writeheader()
        for cluster_id, terms in sorted(cluster_terms.items()):
            for rank, term in enumerate(terms, start=1):
                writer.writerow({"cluster_id": cluster_id, "rank": rank, "term": term})

    (output_dir / "cluster_summary.json").write_text(
        json.dumps(cluster_summary, indent=2),
        encoding="utf-8",
    )

    (output_dir / "cluster_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    if score_rows:
        with (output_dir / "cluster_scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["clusters", "silhouette_score"])
            writer.writeheader()
            writer.writerows(score_rows)


def print_cluster_summary(cluster_summary: list[dict[str, Any]]) -> None:
    """
    Print a compact terminal summary of generated clusters.

    This mirrors the most important parts of cluster_summary.json without
    overwhelming the console: generated label, size, top terms, top topics, and
    key-moment signals.
    """

    for cluster in cluster_summary:
        print()
        print(f"Cluster {cluster['cluster_id']}: {cluster['generated_label']}")
        print(f"  meetings: {cluster['meeting_count']}")
        print(f"  top terms: {', '.join(cluster['top_terms'][:10])}")
        print(
            "  top topics: "
            + ", ".join(f"{topic} ({count})" for topic, count in cluster["top_topics"][:5])
        )
        print(
            "  signals: "
            + ", ".join(f"{signal} ({count})" for signal, count in cluster["key_moment_types"][:5])
        )


def main() -> None:
    """
    Run unsupervised clustering and write review artifacts.

    The CLI supports two evaluation modes:

    - Fixed k: use --clusters N when you want a specific number of buckets for
      presentation or comparison.
    - Auto k: use --auto-k to score a range of k values by silhouette and pick
      the highest-scoring option.

    Either way, the score is only a diagnostic. Final taxonomy decisions should
    include human review of the generated labels and example meetings.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--auto-k", action="store_true")
    parser.add_argument("--min-clusters", type=int, default=4)
    parser.add_argument("--max-clusters", type=int, default=12)
    parser.add_argument("--top-terms", type=int, default=12)
    parser.add_argument("--examples-per-cluster", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    documents = load_meeting_documents(dataset_dir)
    if len(documents) < 2:
        raise SystemExit(f"Need at least two meeting summaries to cluster. Found {len(documents)}.")

    vectorizer, matrix = vectorize_documents(documents, args.max_features)
    clusters = args.clusters
    score_rows: list[dict[str, float | int]] = []
    if args.auto_k:
        clusters, score_rows = choose_cluster_count(
            matrix=matrix,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
            random_state=args.random_state,
        )

    model, labels = fit_clusters(matrix, clusters, args.random_state)
    silhouette = compute_silhouette_score(matrix, list(labels))
    if not score_rows:
        score_rows = [{"clusters": clusters, "silhouette_score": silhouette if silhouette is not None else float("nan")}]

    cluster_terms = top_terms_for_clusters(vectorizer, model, args.top_terms)
    cluster_summary = summarize_clusters(
        documents=documents,
        labels=list(labels),
        cluster_terms=cluster_terms,
        examples_per_cluster=args.examples_per_cluster,
    )
    metrics = {
        "meeting_count": len(documents),
        "clusters": clusters,
        "silhouette_score": silhouette,
        "auto_k": args.auto_k,
        "min_clusters": args.min_clusters if args.auto_k else None,
        "max_clusters": args.max_clusters if args.auto_k else None,
        "random_state": args.random_state,
        "max_features": args.max_features,
    }
    write_outputs(
        output_dir=args.output_dir.resolve(),
        documents=documents,
        labels=list(labels),
        cluster_terms=cluster_terms,
        cluster_summary=cluster_summary,
        metrics=metrics,
        score_rows=score_rows,
    )

    print(f"Loaded {len(documents)} meetings")
    print(f"Generated {clusters} clusters")
    if silhouette is not None:
        print(f"Silhouette score: {silhouette:.4f}")
    else:
        print("Silhouette score: unavailable")
    if args.auto_k:
        print("Candidate k scores:")
        for row in score_rows:
            print(f"  k={int(row['clusters']):2d}  silhouette={float(row['silhouette_score']):.4f}")
    print(f"Wrote outputs to {args.output_dir.resolve()}")
    print_cluster_summary(cluster_summary)


if __name__ == "__main__":
    main()
