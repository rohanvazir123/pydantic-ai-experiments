"""
Compare Take B (TF-IDF/KMeans k=8) vs Take C (HDBSCAN semantic, 26 clusters).
Shows cross-tab, per-B-cluster breakdown, and where Take C splits B's broad clusters.
"""

import json
import pandas as pd

TAKE_B_CLUSTERS = {
    0: "renewal / competitive / pricing",
    1: "outage / incident / failure",
    2: "billing / overage / dispute",
    3: "planning / sprint / launch",
    4: "hipaa / compliance / reporting",
    5: "backup / performance / support response",
    6: "mfa / identity / sso",
    7: "backup / hybrid / recovery / onboarding",
}

BASE = "basics/iprep/meeting-analytics"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    b = pd.read_csv(f"{BASE}/take_b/outputs/meeting_clusters.csv")[["meeting_id", "cluster_id"]]
    b = b.rename(columns={"cluster_id": "b_cluster"})

    c = pd.read_csv(f"{BASE}/take_c/outputs/meeting_themes.csv")[
        ["meeting_id", "primary_theme_id", "primary_theme_title"]
    ]

    with open(f"{BASE}/take_c/outputs/semantic_clusters.json") as f:
        clusters_c: list[dict] = json.load(f)

    return b, c, clusters_c


def cluster_label_c(clusters_c: list[dict], cluster_id: int) -> str:
    for cl in clusters_c:
        if cl["cluster_id"] == cluster_id:
            return cl["theme_title"]
    return "?"


def main() -> None:
    b_df, c_df, clusters_c = load_data()
    merged = b_df.merge(c_df, on="meeting_id", how="inner")
    print(f"\nMeetings matched in both outputs: {len(merged)}/100\n")

    # --- Cross-tab (Take B rows, Take C columns - counts) ---
    crosstab = pd.crosstab(
        merged["b_cluster"],
        merged["primary_theme_id"],
        rownames=["TakeB"],
        colnames=["TakeC"],
    )
    print("=" * 70)
    print("CROSS-TAB: Take B cluster × Take C primary theme (meeting counts)")
    print("=" * 70)
    print(crosstab.to_string())
    print()

    # --- Per-B-cluster breakdown ---
    print("=" * 70)
    print("PER TAKE-B CLUSTER: Which Take C themes does it map to?")
    print("=" * 70)
    for b_id in sorted(TAKE_B_CLUSTERS):
        subset = merged[merged["b_cluster"] == b_id]
        label_b = TAKE_B_CLUSTERS[b_id]
        counts = (
            subset.groupby(["primary_theme_id", "primary_theme_title"])
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        top_c = counts[counts["n"] > 0]
        print(f"\nB-{b_id} [{label_b}]  ({len(subset)} meetings)")
        for _, row in top_c.iterrows():
            bar = "█" * row["n"]
            print(f"  → C-{int(row['primary_theme_id']):02d} {row['primary_theme_title']:<45s} {bar} ({row['n']})")

    # --- Alignment summary ---
    print()
    print("=" * 70)
    print("ALIGNMENT SUMMARY")
    print("=" * 70)

    # For each Take B cluster, top Take C theme capture ratio
    rows = []
    for b_id in sorted(TAKE_B_CLUSTERS):
        subset = merged[merged["b_cluster"] == b_id]
        if subset.empty:
            continue
        top_c_id = subset["primary_theme_id"].value_counts().idxmax()
        top_c_n = subset["primary_theme_id"].value_counts().iloc[0]
        top_c_title = cluster_label_c(clusters_c, int(top_c_id))
        pct = top_c_n / len(subset) * 100
        n_distinct_c = subset["primary_theme_id"].nunique()
        rows.append(
            {
                "B-cluster": f"B-{b_id}",
                "B-label": TAKE_B_CLUSTERS[b_id],
                "meetings": len(subset),
                "C-splits": n_distinct_c,
                "Top-C": f"C-{int(top_c_id):02d} {top_c_title}",
                "top_pct": f"{pct:.0f}%",
            }
        )

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    # --- Where does Take C split Take B? ---
    print()
    print("=" * 70)
    print("SPLIT ANALYSIS: Take B clusters with C-splits > 1")
    print("(These are cases where Take C reveals finer structure inside a B cluster)")
    print("=" * 70)
    for _, row in summary_df.iterrows():
        splits = int(row["C-splits"])
        if splits > 1:
            print(f"  {row['B-cluster']} '{row['B-label']}' → {splits} distinct C themes")

    # --- Overall agreement proxy ---
    # For each meeting, assign its B cluster's "dominant C theme"
    # then check if meeting's own C theme matches it
    b_to_dominant_c: dict[int, int] = {}
    for b_id in sorted(TAKE_B_CLUSTERS):
        subset = merged[merged["b_cluster"] == b_id]
        if not subset.empty:
            b_to_dominant_c[b_id] = int(subset["primary_theme_id"].value_counts().idxmax())

    merged["dominant_c"] = merged["b_cluster"].map(b_to_dominant_c)
    matches = (merged["primary_theme_id"] == merged["dominant_c"]).sum()
    pct_agree = matches / len(merged) * 100
    print(f"\nApprox agreement (meeting's C-theme == B-cluster's dominant C-theme): "
          f"{matches}/{len(merged)} = {pct_agree:.1f}%")
    print("(Higher = B and C are discovering the same primary theme per meeting)\n")


if __name__ == "__main__":
    main()
