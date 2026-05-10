"""
generate_charts.py — Regenerate all analysis charts from pre-exported CSVs.

No database or notebook required. Run this after export_chart_data.py has
produced the CSVs in outputs/chart_data/.

Usage:
    python final_version/generate_charts.py
    python final_version/generate_charts.py --input-dir path/to/chart_data
    python final_version/generate_charts.py --output-dir path/to/charts
    # Or from a Jupyter cell:
    %run final_version/generate_charts.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
# Use Agg only when running as a standalone script (not inside a Jupyter session)
import sys as _sys
if "ipykernel" not in _sys.modules:
    matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "outputs" / "chart_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "charts"

plt.rcParams["figure.figsize"] = (14, 7)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
sns.set_style("whitegrid")

SENTINEL_ORDER = [
    "very-negative", "negative", "mixed-negative",
    "neutral", "mixed-positive", "positive", "very-positive",
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _load(input_dir: Path, name: str) -> pd.DataFrame:
    path = input_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run export_chart_data.py first."
        )
    return pd.read_csv(path)


def _save(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── individual chart functions ─────────────────────────────────────────────────

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
    plt.tight_layout()
    return _save(fig, output_dir, "00_dataset_overview")


def chart_cluster_table(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, max(8, len(df) * 0.35)))
    ax.axis("off")
    col_labels = ["Theme", "Audience", "Phrases", "Primary Meetings"]
    table = ax.table(
        cellText=df[["theme_title", "audience", "phrase_count", "primary_meetings"]].values,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(4)))
    ax.set_title("26 Discovered Themes — ordered by meetings they dominate",
                 fontsize=12, fontweight="bold", pad=20)
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
    plt.tight_layout()
    return _save(fig, output_dir, "02_sentiment_by_calltype")


def chart_theme_sentiment_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    pivot = df.pivot_table(index="theme_title", columns="overall_sentiment",
                           values="meetings", fill_value=0)
    pivot = pivot.reindex(columns=[c for c in SENTINEL_ORDER if c in pivot.columns])
    weights = {s: i for i, s in enumerate(SENTINEL_ORDER)}
    pivot["_score"] = sum(
        pivot.get(c, 0) * weights.get(c, 3) for c in SENTINEL_ORDER
    )
    pivot = pivot.sort_values("_score").drop(columns=["_score"])

    fig, ax = plt.subplots(figsize=(13, max(8, len(pivot) * 0.38)))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="RdYlGn",
                linewidths=0.4, linecolor="white", ax=ax,
                cbar_kws={"label": "Meetings"})
    ax.set_title("Theme × Sentiment Heatmap (primary theme, 26 clusters)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=9)
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
    plt.tight_layout()
    return _save(fig, output_dir, "04_churn_density")


def chart_product_signals(df: pd.DataFrame, output_dir: Path) -> Path:
    x = np.arange(len(df))
    w = 0.28
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w, df["total_meetings"], w, label="Total meetings", color="#aec7e8")
    b2 = ax.bar(x,     df["tech_issues"],    w, label="Technical issues", color="#ff7f0e")
    b3 = ax.bar(x + w, df["churn_signals"],  w, label="Churn signals", color="#d62728")
    for bars in [b1, b2, b3]:
        for bar in bars:
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
    plt.tight_layout()
    return _save(fig, output_dir, "05_product_signals")


def chart_positive_signals(df_praise: pd.DataFrame, df_comply: pd.DataFrame,
                            output_dir: Path) -> Path:
    sent_rank = {s: i for i, s in enumerate(SENTINEL_ORDER)}
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
    detect_total = int(
        df_impact[df_impact["cohort"] == "Mentions Detect"]["meetings"].sum()
    )

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
        "P1 — Feature Gap Requests by Product × Customer Sentiment\n"
        '"Blocked" = P0 churn risk · "Growing" = roadmap investment',
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Feature gap moments")
    ax.legend()
    plt.tight_layout()
    return _save(fig, output_dir, "08_feature_gaps_by_product")


def chart_action_item_owners(df: pd.DataFrame, output_dir: Path) -> Path:
    df = df.head(15)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#d62728" if n >= 10 else "#ff7f0e" if n >= 6 else "#1f77b4"
              for n in df["total_action_items"]]
    bars = ax.barh(
        df["owner"][::-1], df["total_action_items"][::-1],
        color=list(reversed(colors)), height=0.6,
    )
    for bar, v, themes in zip(bars,
                               df["total_action_items"][::-1].values,
                               df["themes_involved"][::-1].values):
        label = f"{int(v)} items · {int(themes)} theme{'s' if themes > 1 else ''}"
        ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9)
    ax.set_title("S3 — Action Item Owners: Who Is Most Overloaded?\n(top 15 across all themes)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Total action items")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    return _save(fig, output_dir, "09_action_item_owners")


# ── orchestrator ───────────────────────────────────────────────────────────────

def generate_all(input_dir: Path, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    specs: list[tuple[str, ...]] = [
        ("call_types", "products"),
        ("clusters",),
        ("sentiment_by_calltype", "avg_sentiment_by_calltype"),
        ("theme_sentiment_heatmap",),
        ("churn_density",),
        ("product_signals",),
        ("praise_by_product", "comply_external_sentiment"),
        ("detect_external_impact", "detect_external_churn"),
        ("feature_gaps_by_product",),
        ("action_item_owners",),
    ]

    fns = [
        lambda d: chart_dataset_overview(d["call_types"], d["products"], output_dir),
        lambda d: chart_cluster_table(d["clusters"], output_dir),
        lambda d: chart_sentiment_calltype(
            d["sentiment_by_calltype"], d["avg_sentiment_by_calltype"], output_dir),
        lambda d: chart_theme_sentiment_heatmap(d["theme_sentiment_heatmap"], output_dir),
        lambda d: chart_churn_density(d["churn_density"], output_dir),
        lambda d: chart_product_signals(d["product_signals"], output_dir),
        lambda d: chart_positive_signals(
            d["praise_by_product"], d["comply_external_sentiment"], output_dir),
        lambda d: chart_detect_external_impact(
            d["detect_external_impact"], d["detect_external_churn"], output_dir),
        lambda d: chart_feature_gaps(d["feature_gaps_by_product"], output_dir),
        lambda d: chart_action_item_owners(d["action_item_owners"], output_dir),
    ]

    # Load only the CSVs we need
    needed = {name for group in specs for name in group}
    data: dict[str, pd.DataFrame] = {name: _load(input_dir, name) for name in needed}

    for fn in fns:
        path = fn(data)
        generated.append(str(path))

    return generated


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate chart PNGs from pre-exported CSVs."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help="Directory containing chart_data CSVs (output of export_chart_data.py)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to write chart PNG files into",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    print(f"Reading CSVs from : {input_dir}")
    print(f"Writing charts to : {output_dir}")
    print("─" * 60)

    generated = generate_all(input_dir, output_dir)
    for p in generated:
        print(f"  {Path(p).name}")
    print("─" * 60)
    print(f"Done. {len(generated)} charts written.")


if __name__ == "__main__":
    main()
