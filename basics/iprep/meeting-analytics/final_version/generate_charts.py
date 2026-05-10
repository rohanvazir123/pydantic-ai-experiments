"""
generate_charts.py — Export chart data from DB to CSVs, then generate PNGs.

Default (exports CSVs then generates PNGs):
    python final_version/generate_charts.py
    python final_version/generate_charts.py --dsn postgresql://user:pass@host/db

Skip DB export (use existing CSVs):
    python final_version/generate_charts.py --no-export

Custom paths:
    python final_version/generate_charts.py --input-dir path/to/csvs --output-dir path/to/pngs

From a Jupyter cell:
    %run final_version/generate_charts.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "outputs" / "chart_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "charts"
DEFAULT_DSN = "postgresql://rag_user:rag_pass@localhost:5434/rag_db"

plt.rcParams["figure.figsize"] = (14, 7)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
sns.set_style("whitegrid")

SENTIMENT_ORDER = [
    "very-negative", "negative", "mixed-negative",
    "neutral", "mixed-positive", "positive", "very-positive",
]

# ── SQL queries (one per CSV file) ────────────────────────────────────────────

QUERIES: dict[str, str] = {
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
        "    sc.theme_title, m.title AS meeting, km.moment_type, "
        "    left(km.text, 200) AS verbatim_quote "
        "FROM meeting_analytics.semantic_clusters sc "
        "JOIN meeting_analytics.semantic_meeting_themes smt "
        "     ON sc.cluster_id = smt.cluster_id AND smt.is_primary = true "
        "JOIN meeting_analytics.meetings m ON smt.meeting_id = m.meeting_id "
        "JOIN meeting_analytics.key_moments km ON smt.meeting_id = km.meeting_id "
        "WHERE length(km.text) > 25 "
        "ORDER BY sc.cluster_id, sc.phrase_count DESC LIMIT 15"
    ),
    "sentiment_by_calltype": (
        "SELECT call_type, "
        "    CASE "
        "        WHEN overall_sentiment IN ('negative','very-negative','mixed-negative') THEN 'Negative' "
        "        WHEN overall_sentiment = 'neutral' THEN 'Neutral' "
        "        ELSE 'Positive' "
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
    "action_items_by_theme": (
        "SELECT theme_title, audience, count(*) AS action_items "
        "FROM meeting_analytics.action_items_by_theme "
        "GROUP BY theme_title, audience ORDER BY action_items DESC"
    ),
    "action_items_by_dept_product": (
        "SELECT abt.audience AS department, unnest(ms.products) AS product, "
        "       count(*) AS action_items "
        "FROM meeting_analytics.action_items_by_theme abt "
        "JOIN meeting_analytics.meeting_summaries ms ON abt.meeting_id = ms.meeting_id "
        "WHERE array_length(ms.products, 1) > 0 "
        "GROUP BY abt.audience, product"
    ),
}


# ── DB export ─────────────────────────────────────────────────────────────────

async def _export_to_csv(dsn: str, csv_dir: Path) -> dict[str, int]:
    import asyncpg
    csv_dir.mkdir(parents=True, exist_ok=True)
    conn = await asyncpg.connect(dsn)
    counts: dict[str, int] = {}
    try:
        for name, sql in QUERIES.items():
            rows = await conn.fetch(sql)
            df = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
            df.to_csv(csv_dir / f"{name}.csv", index=False)
            counts[name] = len(df)
    finally:
        await conn.close()
    return counts


def export_to_csv(dsn: str, csv_dir: Path) -> dict[str, int]:
    """Query DB and write one CSV per chart dataset. Returns {name: row_count}."""
    return asyncio.run(_export_to_csv(dsn, csv_dir))


# ── CSV loader ────────────────────────────────────────────────────────────────

def _load(csv_dir: Path, name: str) -> pd.DataFrame:
    path = csv_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}.\n"
            f"Run with --export-csv to pull fresh data from the database first."
        )
    return pd.read_csv(path)


def _save(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


_DEPT_COLORS: dict[str, tuple[str, str]] = {
    "engineering": ("#1565C0", "#ffffff"),
    "cto":         ("#1565C0", "#ffffff"),
    "product":     ("#2E7D32", "#ffffff"),
    "cpo":         ("#2E7D32", "#ffffff"),
    "sales":       ("#E65100", "#ffffff"),
    "cs":          ("#6A1B9A", "#ffffff"),
    "marketing":   ("#00838F", "#ffffff"),
    "operations":  ("#4E342E", "#ffffff"),
    "leadership":  ("#37474F", "#ffffff"),
}


def _dept_color(name: str) -> tuple[str, str]:
    key = name.lower()
    for kw, colors in _DEPT_COLORS.items():
        if kw in key:
            return colors
    return ("#616161", "#ffffff")


def _tag(fig: plt.Figure, label: str) -> None:
    """Render prominent colored department pills across the top of the figure."""
    depts = [d.strip() for d in label.split("·")]
    # "For:" prefix
    fig.text(
        0.01, 0.993, "For:",
        ha="left", va="top", fontsize=10, color="#444444", fontweight="bold",
        transform=fig.transFigure,
    )
    x = 0.065           # start after "For:" label
    char_w = 0.0088     # approx figure-units per char at fontsize 11
    gap = 0.016         # gap between pills
    for dept in depts:
        bg, fg = _dept_color(dept)
        fig.text(
            x, 0.996, dept,
            ha="left", va="top", fontsize=11, color=fg, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=bg,
                      edgecolor="none", alpha=0.95),
            transform=fig.transFigure,
        )
        x += len(dept) * char_w + gap


# ── chart functions ───────────────────────────────────────────────────────────

def chart_dataset_overview(df_call: pd.DataFrame, df_prod: pd.DataFrame,
                            output_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(df_call["call_type"], df_call["meetings"],
            color=["#d62728", "#1f77b4", "#7f7f7f"], width=0.5)
    for i, v in enumerate(df_call["meetings"]):
        ax1.text(i, v + 0.4, str(v), ha="center", fontweight="bold")
    ax1.set_title("Call Type Breakdown", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Meetings")
    ax1.set_ylim(0, df_call["meetings"].max() * 1.2)

    colors_p = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    ax2.bar(df_prod["product"], df_prod["meetings"],
            color=colors_p[: len(df_prod)], width=0.5)
    for i, v in enumerate(df_prod["meetings"]):
        ax2.text(i, v + 0.4, str(v), ha="center", fontweight="bold")
    ax2.set_title("Product Mentions per Meeting", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Meetings mentioning product")
    ax2.set_ylim(0, df_prod["meetings"].max() * 1.2)

    fig.suptitle("AegisCloud — 100 Customer Meetings", fontsize=14, fontweight="bold")
    _tag(fig, "All Leadership")
    plt.tight_layout()
    return _save(fig, output_dir, "00_dataset_overview")


def chart_cluster_table(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, max(8, len(df) * 0.35)))
    ax.axis("off")
    table = ax.table(
        cellText=df[["theme_title", "audience", "phrase_count", "primary_meetings"]].values,
        colLabels=["Theme", "Audience", "Phrases", "Primary Meetings"],
        cellLoc="left", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(4)))
    ax.set_title("26 Discovered Themes — ordered by meetings they dominate",
                 fontsize=12, fontweight="bold", pad=20)
    _tag(fig, "All Leadership")
    plt.tight_layout()
    return _save(fig, output_dir, "01_cluster_table")


def chart_sentiment_calltype(df_sent: pd.DataFrame, df_avg: pd.DataFrame,
                              output_dir: Path) -> Path:
    pivot = df_sent.pivot_table(index="call_type", columns="bucket",
                                values="meetings", fill_value=0)
    for col in ["Negative", "Neutral", "Positive"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["Negative", "Neutral", "Positive"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    bottom = np.zeros(len(pivot))
    for bucket, color in [("Negative", "#d62728"), ("Neutral", "#aec7e8"), ("Positive", "#2ca02c")]:
        vals = pivot[bucket].values
        bars = ax1.bar(pivot.index, vals, bottom=bottom, label=bucket, color=color, width=0.5)
        for bar, v, bot in zip(bars, vals, bottom):
            if v > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bot + v / 2, str(int(v)),
                         ha="center", va="center", color="white", fontweight="bold", fontsize=11)
        bottom += vals
    ax1.set_title("Sentiment Distribution by Call Type", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Meetings")
    ax1.legend()

    score_colors = ["#d62728" if s < 3.0 else "#ff9800" if s < 3.5 else "#2ca02c"
                    for s in df_avg["avg_score"]]
    bars2 = ax2.bar(df_avg["call_type"], df_avg["avg_score"].astype(float),
                    color=score_colors, width=0.5)
    for bar, val in zip(bars2, df_avg["avg_score"].astype(float)):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontweight="bold")
    ax2.axhline(y=3.0, color="grey", linestyle="--", linewidth=1, label="Neutral threshold (3.0)")
    ax2.set_title("Average Sentiment Score by Call Type\n(1 = most negative, 5 = most positive)",
                  fontsize=13, fontweight="bold")
    ax2.set_ylabel("Avg sentiment score")
    ax2.set_ylim(0, 5.2)
    ax2.legend(fontsize=9)
    _tag(fig, "All Leadership")
    plt.tight_layout()
    return _save(fig, output_dir, "02_sentiment_by_calltype")


def chart_theme_sentiment_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    pivot = df.pivot_table(index="theme_title", columns="overall_sentiment",
                           values="meetings", fill_value=0)
    pivot = pivot.reindex(columns=[c for c in SENTIMENT_ORDER if c in pivot.columns])
    weights = {s: i for i, s in enumerate(SENTIMENT_ORDER)}
    pivot["_score"] = sum(pivot.get(c, 0) * weights.get(c, 3) for c in SENTIMENT_ORDER)
    pivot = pivot.sort_values("_score").drop(columns=["_score"]).astype(int)

    fig, ax = plt.subplots(figsize=(13, max(8, len(pivot) * 0.38)))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="RdYlGn",
                linewidths=0.4, linecolor="white", ax=ax,
                cbar_kws={"label": "Meetings"})
    ax.set_title("Theme × Sentiment Heatmap (primary theme, 26 clusters)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=9)
    _tag(fig, "All Leadership")
    plt.tight_layout()
    return _save(fig, output_dir, "03_theme_sentiment_heatmap")


def chart_churn_density(df: pd.DataFrame, output_dir: Path) -> Path:
    df = df.copy()
    df["per_meeting"] = df["per_meeting"].astype(float)
    df_s = df.sort_values("per_meeting", ascending=True)

    fig, ax = plt.subplots(figsize=(11, max(6, len(df_s) * 0.38)))
    colors = ["#d62728" if v >= 0.9 else "#ff7f0e" if v >= 0.5 else "#aec7e8"
              for v in df_s["per_meeting"]]
    bars = ax.barh(df_s["theme_title"], df_s["per_meeting"], color=colors, height=0.6)
    for bar, val in zip(bars, df_s["per_meeting"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)
    mean_val = df["per_meeting"].mean()
    ax.axvline(x=mean_val, color="#555", linestyle="--", linewidth=1)
    ax.text(mean_val + 0.01, 0.5, f"avg {mean_val:.2f}",
            color="#555", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_title("Insight 3.1 — Churn Signal Density by Theme (signals per meeting)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Churn signals per meeting")
    ax.tick_params(axis="y", labelsize=9)
    _tag(fig, "Sales & CS")
    plt.tight_layout()
    return _save(fig, output_dir, "04_churn_density")


def chart_product_signals(df: pd.DataFrame, output_dir: Path) -> Path:
    x = np.arange(len(df))
    w = 0.28
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w, df["total_meetings"], w, label="Total meetings", color="#aec7e8")
    b2 = ax.bar(x,     df["tech_issues"],    w, label="Technical issues", color="#ff7f0e")
    b3 = ax.bar(x + w, df["churn_signals"],  w, label="Churn signals", color="#d62728")
    for group in [b1, b2, b3]:
        for bar in group:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        str(int(h)), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(df["product"], fontsize=12)
    ax.set_title("Insight 3.3 — Technical Issues & Churn Signals by Product",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.legend()
    _tag(fig, "Engineering · Product")
    plt.tight_layout()
    return _save(fig, output_dir, "05_product_signals")


def chart_positive_signals(df_praise: pd.DataFrame, df_comply: pd.DataFrame,
                            output_dir: Path) -> Path:
    sent_rank = {s: i for i, s in enumerate(SENTIMENT_ORDER)}
    df_comply = df_comply.copy()
    df_comply["_r"] = df_comply["overall_sentiment"].map(sent_rank).fillna(99)
    df_comply = df_comply.sort_values("_r").drop(columns=["_r"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors_pr = ["#4CAF50" if p == "Comply" else "#81C784" for p in df_praise["product"]]
    bars1 = ax1.bar(df_praise["product"], df_praise["praise_moments"],
                    color=colors_pr, width=0.5)
    for bar, v in zip(bars1, df_praise["praise_moments"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.2, str(v),
                 ha="center", va="bottom", fontweight="bold")
    ax1.set_title("Praise Moments by Product", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Praise key moments")
    ax1.set_ylim(0, df_praise["praise_moments"].max() * 1.2)

    sent_colors = ["#d62728" if "negative" in s and "mixed" not in s
                   else "#ff9800" if "mixed" in s
                   else "#4CAF50" if "positive" in s
                   else "#9E9E9E" for s in df_comply["overall_sentiment"]]
    bars2 = ax2.barh(df_comply["overall_sentiment"], df_comply["meetings"],
                     color=sent_colors, height=0.5)
    for bar, v in zip(bars2, df_comply["meetings"]):
        ax2.text(v + 0.05, bar.get_y() + bar.get_height() / 2,
                 str(v), va="center", fontweight="bold")
    ax2.set_title("Comply — External Meeting Sentiment\n(renewal & account calls only)",
                  fontsize=13, fontweight="bold")
    ax2.set_xlabel("Meetings")
    fig.suptitle("Insight 3.4 — Positive Signals: The Comply Counter-Narrative",
                 fontsize=14, fontweight="bold")
    _tag(fig, "Product · Marketing · Sales & CS")
    plt.tight_layout()
    return _save(fig, output_dir, "06_positive_signals")


def chart_detect_external_impact(df_impact: pd.DataFrame, df_churn: pd.DataFrame,
                                  output_dir: Path) -> Path:
    pv = df_impact.pivot_table(index="cohort", columns="bucket",
                                values="meetings", fill_value=0)
    for col in ["Negative", "Neutral", "Positive"]:
        if col not in pv.columns:
            pv[col] = 0
    pv = pv[["Negative", "Neutral", "Positive"]]

    churn_n = int(df_churn.iloc[0]["meetings_with_churn"])
    detect_total = int(df_impact[df_impact["cohort"] == "Mentions Detect"]["meetings"].sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    bottom = np.zeros(len(pv))
    for bucket, color in [("Negative", "#d62728"), ("Neutral", "#aec7e8"), ("Positive", "#2ca02c")]:
        vals = pv[bucket].values
        bars = ax1.bar(pv.index, vals, bottom=bottom, label=bucket, color=color, width=0.5)
        for bar, v, bot in zip(bars, vals, bottom):
            if v > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bot + v / 2, str(int(v)),
                         ha="center", va="center", color="white", fontweight="bold")
        bottom += vals
    ax1.set_title("External Meetings: Detect-tagged vs Detect-free\n(renewal · account · commercial)",
                  fontweight="bold")
    ax1.set_ylabel("Meetings")
    ax1.legend()

    ax2.pie(
        [churn_n, detect_total - churn_n],
        labels=[f"Has churn signal\n({churn_n})", f"No churn signal\n({detect_total - churn_n})"],
        colors=["#d62728", "#aec7e8"],
        startangle=90, autopct="%1.0f%%", textprops={"fontsize": 11},
    )
    ax2.set_title(f"Of {detect_total} external Detect meetings\nhow many carry a churn signal?",
                  fontweight="bold")
    fig.suptitle("E3/R4 — Detect Outage Impact on External (Renewal + Account) Meetings",
                 fontsize=13, fontweight="bold")
    _tag(fig, "Engineering (CTO) · Sales")
    plt.tight_layout()
    return _save(fig, output_dir, "07_detect_external_impact")


def chart_feature_gaps(df: pd.DataFrame, output_dir: Path) -> Path:
    pv = df.pivot_table(index="product", columns="sentiment_bucket",
                         values="feature_gaps", fill_value=0)
    for col in ["Blocked (negative)", "Neutral", "Growing (positive)"]:
        if col not in pv.columns:
            pv[col] = 0
    pv = pv[["Blocked (negative)", "Neutral", "Growing (positive)"]]

    x = np.arange(len(pv))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (col, color) in enumerate([
        ("Blocked (negative)", "#d62728"),
        ("Neutral", "#aec7e8"),
        ("Growing (positive)", "#2ca02c"),
    ]):
        bars = ax.bar(x + (i - 1) * w, pv[col], w, label=col, color=color)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1, str(int(h)),
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(pv.index, fontsize=12)
    ax.set_title(
        'P1 — Feature Gap Requests by Product × Customer Sentiment\n'
        '"Blocked" = P0 churn risk · "Growing" = roadmap investment',
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Feature gap moments")
    ax.legend()
    _tag(fig, "Product (CPO)")
    plt.tight_layout()
    return _save(fig, output_dir, "08_feature_gaps_by_product")


def chart_action_item_owners(df_theme: pd.DataFrame, df_dept_product: pd.DataFrame,
                              output_dir: Path) -> Path:
    # Left panel: action items per theme (no owner names), sorted ascending for barh
    theme_totals = (
        df_theme.groupby("theme_title")["action_items"].sum()
        .sort_values(ascending=True)
    )

    # Right panel: department × product stacked bars
    product_colors = {"Detect": "#2196F3", "Comply": "#4CAF50",
                      "Protect": "#FF9800", "Identity": "#9C27B0"}
    pivot_dp = (
        df_dept_product.pivot_table(index="department", columns="product",
                                     values="action_items", fill_value=0)
        .astype(int)
    )
    # Sort departments by total descending (ascending for barh top = most)
    pivot_dp = pivot_dp.loc[pivot_dp.sum(axis=1).sort_values(ascending=True).index]

    n_themes = len(theme_totals)
    n_depts = len(pivot_dp)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(8, max(n_themes, n_depts) * 0.38)),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # ── Left: theme bar chart ──────────────────────────────────────────────────
    theme_color = "#5c85d6"
    bars = ax1.barh(theme_totals.index, theme_totals.values, color=theme_color, height=0.6)
    for bar, val in zip(bars, theme_totals.values):
        ax1.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                 str(int(val)), va="center", fontsize=8)
    ax1.set_title("Action Items by Theme", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Action items")
    ax1.tick_params(axis="y", labelsize=9)

    # ── Right: department × product stacked bars ───────────────────────────────
    left = np.zeros(n_depts)
    for col in pivot_dp.columns:
        color = product_colors.get(col, "#9E9E9E")
        vals = pivot_dp[col].values.astype(float)
        bars2 = ax2.barh(pivot_dp.index, vals, left=left, color=color, label=col, height=0.6)
        for bar, v, lft in zip(bars2, vals, left):
            if v >= 2:
                ax2.text(lft + v / 2, bar.get_y() + bar.get_height() / 2,
                         str(int(v)), ha="center", va="center",
                         color="white", fontsize=8, fontweight="bold")
        left += vals
    for y_idx, total in enumerate(pivot_dp.sum(axis=1)):
        if total > 0:
            ax2.text(total + 0.2, y_idx, str(int(total)), va="center", fontsize=8)
    ax2.set_title("Action Items by Department × Product", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Action items")
    ax2.tick_params(axis="y", labelsize=9)
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.7)

    fig.suptitle("S3 — Action Item Workload: By Theme and Department",
                 fontsize=14, fontweight="bold")
    _tag(fig, "Operations · Engineering · CS · Sales")
    plt.tight_layout()
    return _save(fig, output_dir, "09_action_item_owners")


# ── orchestrator ──────────────────────────────────────────────────────────────

CHART_SPECS: list[tuple[tuple[str, ...], object]] = [
    (("call_types", "products"),              lambda d, o: chart_dataset_overview(d["call_types"], d["products"], o)),
    (("clusters",),                           lambda d, o: chart_cluster_table(d["clusters"], o)),
    (("sentiment_by_calltype",
      "avg_sentiment_by_calltype"),           lambda d, o: chart_sentiment_calltype(d["sentiment_by_calltype"], d["avg_sentiment_by_calltype"], o)),
    (("theme_sentiment_heatmap",),            lambda d, o: chart_theme_sentiment_heatmap(d["theme_sentiment_heatmap"], o)),
    (("churn_density",),                      lambda d, o: chart_churn_density(d["churn_density"], o)),
    (("product_signals",),                    lambda d, o: chart_product_signals(d["product_signals"], o)),
    (("praise_by_product",
      "comply_external_sentiment"),           lambda d, o: chart_positive_signals(d["praise_by_product"], d["comply_external_sentiment"], o)),
    (("detect_external_impact",
      "detect_external_churn"),              lambda d, o: chart_detect_external_impact(d["detect_external_impact"], d["detect_external_churn"], o)),
    (("feature_gaps_by_product",),            lambda d, o: chart_feature_gaps(d["feature_gaps_by_product"], o)),
    (("action_items_by_theme",
      "action_items_by_dept_product"),         lambda d, o: chart_action_item_owners(d["action_items_by_theme"], d["action_items_by_dept_product"], o)),
]


def generate_all(csv_dir: Path, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    needed = {name for keys, _ in CHART_SPECS for name in keys}
    data = {name: _load(csv_dir, name) for name in needed}
    return [str(fn(data, output_dir)) for _, fn in CHART_SPECS]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate chart PNGs from CSVs. Use --export-csv to pull fresh data from the DB first."
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Skip DB export and use existing CSVs in --input-dir.",
    )
    parser.add_argument(
        "--dsn", default=DEFAULT_DSN,
        help="PostgreSQL DSN (used unless --no-export is set).",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help="Directory containing chart_data CSVs.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to write chart PNG files into.",
    )
    args = parser.parse_args()

    csv_dir = args.input_dir.resolve()
    chart_dir = args.output_dir.resolve()

    sep = "-" * 60
    if not args.no_export:
        print(f"Exporting CSVs from DB -> {csv_dir}")
        print(sep)
        counts = export_to_csv(args.dsn, csv_dir)
        for name, n in counts.items():
            print(f"  {name:<35s}: {n} rows")
        print(sep)

    print(f"Generating charts from {csv_dir}")
    print(f"Writing PNGs to        {chart_dir}")
    print(sep)
    generated = generate_all(csv_dir, chart_dir)
    for p in generated:
        print(f"  {Path(p).name}")
    print(sep)
    print(f"Done. {len(generated)} charts written to {chart_dir}")


if __name__ == "__main__":
    main()
