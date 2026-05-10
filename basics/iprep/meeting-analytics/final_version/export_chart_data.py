"""
export_chart_data.py — Export all notebook chart data to CSV.

Runs every SQL query used in meeting_analytics.ipynb and saves the result
to final_version/outputs/chart_data/*.csv.  The CSVs can then be used to
regenerate charts without a live database connection.

Usage:
    python final_version/export_chart_data.py          # uses default DSN
    python final_version/export_chart_data.py --dsn postgresql://...
    # Or from a Jupyter cell:
    %run final_version/export_chart_data.py
"""

from __future__ import annotations

import asyncio
import argparse
from pathlib import Path

import asyncpg
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "chart_data"
DEFAULT_DSN = "postgresql://rag_user:rag_pass@localhost:5434/rag_db"

# ── SQL definitions ────────────────────────────────────────────────────────────

QUERIES: dict[str, str] = {
    # Dataset overview
    "call_types": (
        "SELECT call_type, count(DISTINCT meeting_id) AS meetings "
        "FROM meeting_analytics.semantic_meeting_themes "
        "WHERE is_primary = true GROUP BY call_type ORDER BY meetings DESC"
    ),
    "products": (
        "SELECT unnest(products) AS product, count(*) AS meetings "
        "FROM meeting_analytics.meeting_summaries "
        "GROUP BY product ORDER BY meetings DESC"
    ),
    # Task 1 — clustering output
    "clusters": (
        "SELECT sc.theme_title, sc.audience, sc.phrase_count, "
        "       count(DISTINCT smt.meeting_id) AS primary_meetings "
        "FROM meeting_analytics.semantic_clusters sc "
        "JOIN meeting_analytics.semantic_meeting_themes smt ON sc.cluster_id = smt.cluster_id "
        "WHERE smt.is_primary = true "
        "GROUP BY sc.theme_title, sc.audience, sc.phrase_count "
        "ORDER BY primary_meetings DESC"
    ),
    "examples": (
        "SELECT DISTINCT ON (sc.cluster_id) "
        "    sc.theme_title, "
        "    m.title AS meeting, "
        "    km.moment_type, "
        "    left(km.text, 200) AS verbatim_quote "
        "FROM meeting_analytics.semantic_clusters sc "
        "JOIN meeting_analytics.semantic_meeting_themes smt "
        "     ON sc.cluster_id = smt.cluster_id AND smt.is_primary = true "
        "JOIN meeting_analytics.meetings m ON smt.meeting_id = m.meeting_id "
        "JOIN meeting_analytics.key_moments km ON smt.meeting_id = km.meeting_id "
        "WHERE length(km.text) > 25 "
        "ORDER BY sc.cluster_id, sc.phrase_count DESC "
        "LIMIT 15"
    ),
    # Task 2 — sentiment analysis
    "sentiment_by_calltype": (
        "SELECT call_type, "
        "    CASE "
        "        WHEN overall_sentiment IN ('negative','very-negative','mixed-negative') THEN 'Negative' "
        "        WHEN overall_sentiment = 'neutral' THEN 'Neutral' "
        "        WHEN overall_sentiment IN ('positive','very-positive','mixed-positive') THEN 'Positive' "
        "        ELSE 'Unknown' "
        "    END AS bucket, "
        "    count(*) AS meetings "
        "FROM meeting_analytics.semantic_meeting_themes "
        "WHERE is_primary = true AND overall_sentiment IS NOT NULL "
        "GROUP BY call_type, bucket ORDER BY call_type, meetings DESC"
    ),
    "avg_sentiment_by_calltype": (
        "SELECT call_type, "
        "    round(avg(sentiment_score)::numeric, 2) AS avg_score, "
        "    count(*) AS meetings "
        "FROM meeting_analytics.semantic_meeting_themes "
        "WHERE is_primary = true "
        "GROUP BY call_type ORDER BY avg_score ASC"
    ),
    "theme_sentiment_heatmap": (
        "SELECT sc.theme_title, smt.overall_sentiment, count(*) AS meetings "
        "FROM meeting_analytics.semantic_meeting_themes smt "
        "JOIN meeting_analytics.semantic_clusters sc ON smt.cluster_id = sc.cluster_id "
        "WHERE smt.is_primary = true AND smt.overall_sentiment IS NOT NULL "
        "GROUP BY sc.theme_title, smt.overall_sentiment"
    ),
    # Task 3 — insights
    "churn_density": (
        "SELECT sc.theme_title, "
        "       count(DISTINCT smt.meeting_id) AS meetings, "
        "       count(km.moment_index) AS churn_signals, "
        "       round(count(km.moment_index)::numeric / NULLIF(count(DISTINCT smt.meeting_id),0), 2) AS per_meeting "
        "FROM meeting_analytics.semantic_clusters sc "
        "JOIN meeting_analytics.semantic_meeting_themes smt "
        "     ON sc.cluster_id = smt.cluster_id AND smt.is_primary = true "
        "LEFT JOIN meeting_analytics.key_moments km "
        "     ON smt.meeting_id = km.meeting_id AND km.moment_type = 'churn_signal' "
        "GROUP BY sc.theme_title HAVING count(km.moment_index) > 0 "
        "ORDER BY per_meeting DESC"
    ),
    "watchlist": (
        "SELECT m.meeting_id, m.title, m.organizer_email, "
        "       sc.theme_title AS primary_theme, smt.call_type, "
        "       smt.overall_sentiment, round(smt.sentiment_score::numeric, 1) AS score, "
        "       count(km.moment_index) AS churn_signals "
        "FROM meeting_analytics.meetings m "
        "JOIN meeting_analytics.semantic_meeting_themes smt "
        "     ON m.meeting_id = smt.meeting_id AND smt.is_primary = true "
        "JOIN meeting_analytics.semantic_clusters sc ON smt.cluster_id = sc.cluster_id "
        "JOIN meeting_analytics.key_moments km "
        "     ON m.meeting_id = km.meeting_id AND km.moment_type = 'churn_signal' "
        "WHERE smt.overall_sentiment IN ('negative','very-negative','mixed-negative') "
        "GROUP BY m.meeting_id, m.title, m.organizer_email, sc.theme_title, "
        "         smt.call_type, smt.overall_sentiment, smt.sentiment_score "
        "ORDER BY churn_signals DESC, smt.sentiment_score ASC"
    ),
    "product_signals": (
        "SELECT p.product, "
        "       count(DISTINCT p.meeting_id) AS total_meetings, "
        "       count(km_t.moment_index) AS tech_issues, "
        "       count(km_c.moment_index) AS churn_signals "
        "FROM (SELECT DISTINCT unnest(products) AS product, meeting_id "
        "      FROM meeting_analytics.meeting_summaries WHERE products <> '{}') p "
        "LEFT JOIN meeting_analytics.key_moments km_t "
        "     ON p.meeting_id = km_t.meeting_id AND km_t.moment_type = 'technical_issue' "
        "LEFT JOIN meeting_analytics.key_moments km_c "
        "     ON p.meeting_id = km_c.meeting_id AND km_c.moment_type = 'churn_signal' "
        "GROUP BY p.product ORDER BY tech_issues DESC"
    ),
    "praise_by_product": (
        "SELECT p.product, count(km.moment_index) AS praise_moments "
        "FROM (SELECT DISTINCT unnest(products) AS product, meeting_id "
        "      FROM meeting_analytics.meeting_summaries WHERE products <> '{}') p "
        "JOIN meeting_analytics.key_moments km "
        "     ON p.meeting_id = km.meeting_id AND km.moment_type = 'praise' "
        "GROUP BY p.product ORDER BY praise_moments DESC"
    ),
    "comply_external_sentiment": (
        "SELECT overall_sentiment, count(*) AS meetings "
        "FROM meeting_analytics.semantic_meeting_themes "
        "WHERE 'Comply' = ANY(products) AND call_type = 'external' AND is_primary = true "
        "GROUP BY overall_sentiment ORDER BY meetings DESC"
    ),
    # Leadership questions (E3, P1, S3)
    "detect_external_impact": (
        "SELECT "
        "  CASE WHEN 'Detect' = ANY(smt.products) THEN 'Mentions Detect' ELSE 'No Detect' END AS cohort, "
        "  CASE "
        "    WHEN smt.overall_sentiment IN ('negative','very-negative','mixed-negative') THEN 'Negative' "
        "    WHEN smt.overall_sentiment = 'neutral' THEN 'Neutral' "
        "    ELSE 'Positive' "
        "  END AS bucket, "
        "  count(*) AS meetings "
        "FROM meeting_analytics.semantic_meeting_themes smt "
        "WHERE smt.call_type = 'external' AND smt.is_primary = true "
        "GROUP BY cohort, bucket"
    ),
    "detect_external_churn": (
        "SELECT count(DISTINCT km.meeting_id) AS meetings_with_churn "
        "FROM meeting_analytics.key_moments km "
        "JOIN meeting_analytics.semantic_meeting_themes smt "
        "  ON km.meeting_id = smt.meeting_id AND smt.is_primary = true "
        "WHERE smt.call_type = 'external' "
        "  AND 'Detect' = ANY(smt.products) "
        "  AND km.moment_type = 'churn_signal'"
    ),
    "feature_gaps_by_product": (
        "SELECT p.product, "
        "  CASE "
        "    WHEN ms.overall_sentiment IN ('negative','very-negative','mixed-negative') THEN 'Blocked (negative)' "
        "    WHEN ms.overall_sentiment = 'neutral' THEN 'Neutral' "
        "    ELSE 'Growing (positive)' "
        "  END AS sentiment_bucket, "
        "  count(km.moment_index) AS feature_gaps "
        "FROM (SELECT DISTINCT unnest(products) AS product, meeting_id "
        "      FROM meeting_analytics.meeting_summaries WHERE products <> '{}') p "
        "JOIN meeting_analytics.key_moments km "
        "  ON p.meeting_id = km.meeting_id AND km.moment_type = 'feature_gap' "
        "JOIN meeting_analytics.meeting_summaries ms ON p.meeting_id = ms.meeting_id "
        "GROUP BY p.product, sentiment_bucket"
    ),
    "action_item_owners": (
        "SELECT owner, "
        "  count(*) AS total_action_items, "
        "  count(DISTINCT theme_title) AS themes_involved "
        "FROM meeting_analytics.action_items_by_theme "
        "GROUP BY owner ORDER BY total_action_items DESC LIMIT 15"
    ),
}


# ── Core ───────────────────────────────────────────────────────────────────────

async def export_all(dsn: str, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    conn = await asyncpg.connect(dsn)
    saved: dict[str, int] = {}
    try:
        for name, sql in QUERIES.items():
            rows = await conn.fetch(sql)
            df = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
            path = output_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            saved[name] = len(df)
    finally:
        await conn.close()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all notebook chart data to CSV.")
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="PostgreSQL DSN")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write CSVs into")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    print(f"Exporting chart data → {output_dir}")
    print("─" * 60)

    saved = asyncio.run(export_all(args.dsn, output_dir))

    for name, n in saved.items():
        print(f"  {name:<35s}: {n} rows → {name}.csv")
    print("─" * 60)
    print(f"Done. {len(saved)} CSV files written to {output_dir}")


if __name__ == "__main__":
    main()
