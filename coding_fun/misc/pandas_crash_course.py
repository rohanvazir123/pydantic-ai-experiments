"""
Pandas crash course — using Baby Names (SSA) and OWID COVID datasets.

Run download.py first.

Each section demonstrates a core concept and ends with a challenge.
Run with --solutions to print all answers.

Usage:
    python pandas_crash_course.py
    python pandas_crash_course.py --section 4
    python pandas_crash_course.py --solutions
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# Force UTF-8 output so Unicode chars in print statements work on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")  # suppress pandas FutureWarnings for cleaner output

try:
    import pandas as pd
except ImportError:
    sys.exit("Install pandas: pip install pandas")

DATASETS = Path(__file__).parent / "datasets"
BABY_DIR = DATASETS / "baby_names"
COVID_CSV = DATASETS / "owid_covid.csv"


def _check_data() -> None:
    missing = []
    if not BABY_DIR.exists():
        missing.append("baby_names/ (run download.py)")
    if not COVID_CSV.exists():
        missing.append("owid_covid.csv (run download.py)")
    if missing:
        sys.exit("Missing data:\n  " + "\n  ".join(missing))


def section(n: int, title: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  SECTION {n}: {title}")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────
# SECTION 1 — Loading data
# ─────────────────────────────────────────────────────────────
def s1_loading() -> tuple[pd.DataFrame, pd.DataFrame]:
    section(1, "Loading data — read_csv, concat, dtypes, memory")

    # ── 1a: single CSV ────────────────────────────────────────
    print("1a) Read one year file (no header → supply names manually)")
    df23 = pd.read_csv(
        BABY_DIR / "yob2023.txt",
        header=None,
        names=["name", "sex", "count"],
    )
    print(df23.head(3))
    print(f"Shape: {df23.shape}  dtypes: name={df23['name'].dtype}, count={df23['count'].dtype}")

    # ── 1b: concat all years ──────────────────────────────────
    print("\n1b) Concat all year files into one DataFrame")
    frames = []
    for f in sorted(BABY_DIR.glob("yob*.txt")):
        year = int(f.stem[3:])
        tmp = pd.read_csv(f, header=None, names=["name", "sex", "count"])
        tmp["year"] = year
        frames.append(tmp)
    baby = pd.concat(frames, ignore_index=True)
    print(f"Combined shape: {baby.shape}")
    print(f"Years: {baby['year'].min()} – {baby['year'].max()}")
    mem = baby.memory_usage(deep=True).sum() / 1e6
    print(f"Memory: {mem:.1f} MB")

    # ── 1c: large CSV with dtypes & dates ────────────────────
    print("\n1c) Read OWID COVID CSV (subset of columns)")
    COLS = ["iso_code", "location", "date", "new_cases", "new_deaths",
            "total_cases", "total_deaths", "population"]
    covid = pd.read_csv(
        COVID_CSV,
        usecols=COLS,
        parse_dates=["date"],
        dtype={"iso_code": "category", "location": "category"},
    )
    print(f"Shape: {covid.shape}")
    print(covid.dtypes)
    mem2 = covid.memory_usage(deep=True).sum() / 1e6
    print(f"Memory: {mem2:.1f} MB")

    print("""
KEY POINTS
  pd.read_csv(path, header=None, names=[...])   → supply column names
  pd.concat(frames, ignore_index=True)          → stack DataFrames vertically
  usecols=[...]                                 → load only needed columns (saves RAM)
  parse_dates=["col"]                           → auto-parse to datetime64
  dtype={"col": "category"}                     → use categorical for low-cardinality strings
""")

    print("""CHALLENGE 1
  Load the COVID CSV but keep ONLY rows for G7 countries
  (USA, GBR, DEU, FRA, ITA, JPN, CAN) using the iso_code column.
  What is the total new_cases sum for each country?
  (Solution at bottom)
""")
    return baby, covid


# ─────────────────────────────────────────────────────────────
# SECTION 2 — Exploration
# ─────────────────────────────────────────────────────────────
def s2_exploration(baby: pd.DataFrame, covid: pd.DataFrame) -> None:
    section(2, "Exploration — info, describe, value_counts, nunique")

    print("2a) baby.info()")
    baby.info(memory_usage="deep")

    print("\n2b) covid.describe() — numeric summary")
    print(covid[["new_cases", "new_deaths", "population"]].describe().round(0))

    print("\n2c) value_counts — top locations in COVID data")
    print(covid["location"].value_counts().head(8))

    print("\n2d) nulls per column")
    nulls = covid.isnull().sum()
    print(nulls[nulls > 0].sort_values(ascending=False).head(8))

    print("\n2e) nunique")
    print(f"Unique countries: {covid['location'].nunique()}")
    print(f"Date range: {covid['date'].min().date()} → {covid['date'].max().date()}")

    print("""
KEY POINTS
  df.info()              → column types + non-null counts
  df.describe()          → count, mean, std, min, quartiles, max
  df["col"].value_counts() → frequency table, sorted descending
  df.isnull().sum()      → null count per column
  df["col"].nunique()    → count of distinct values
""")

    print("""CHALLENGE 2
  In the baby names data, how many names appear in EVERY year from 1880 to 2023?
  Hint: groupby name, count distinct years, filter where count == number of years in range.
  (Solution at bottom)
""")


# ─────────────────────────────────────────────────────────────
# SECTION 3 — Selection
# ─────────────────────────────────────────────────────────────
def s3_selection(baby: pd.DataFrame, covid: pd.DataFrame) -> None:
    section(3, "Selection — loc, iloc, boolean indexing, query()")

    print("3a) Boolean indexing — female names in 2023")
    f23 = baby[(baby["year"] == 2023) & (baby["sex"] == "F")]
    print(f"Female rows in 2023: {len(f23)}")
    print(f"Top 5:\n{f23.nlargest(5, 'count')[['name','count']].to_string(index=False)}")

    print("\n3b) .query() — same as above, more readable")
    f23q = baby.query("year == 2023 and sex == 'F'")
    print(f"Same result: {len(f23q)} rows")

    print("\n3c) .loc — label-based selection")
    emma = baby.loc[(baby["name"] == "Emma") & (baby["sex"] == "F"), ["year", "count"]]
    print(f"Emma (F) across all years — last 5:\n{emma.tail(5).to_string(index=False)}")

    print("\n3d) .iloc — position-based selection")
    print("First 3 rows, first 2 cols:")
    print(baby.iloc[:3, :2])

    print("\n3e) isin()")
    names = ["Emma", "Olivia", "Ava"]
    subset = baby.query("year == 2023 and sex == 'F'").loc[baby["name"].isin(names)]
    print(subset[["name", "count"]].sort_values("count", ascending=False))

    print("""
KEY POINTS
  df[mask]                → boolean mask (Series of True/False)
  df.query("expr")        → string-based; cleaner for complex conditions
  df.loc[row_mask, cols]  → label-based; cols is list of column names
  df.iloc[rows, cols]     → position-based; rows/cols are integers/slices
  df["col"].isin([...])   → True where value is in the list
  ~ mask                  → NOT (invert a boolean mask)
""")

    print("""CHALLENGE 3
  Find all COVID rows where new_cases > 100_000 AND the date is in 2021.
  Which location appears most often in that filtered set?
  (Solution at bottom)
""")


# ─────────────────────────────────────────────────────────────
# SECTION 4 — Aggregation
# ─────────────────────────────────────────────────────────────
def s4_aggregation(baby: pd.DataFrame, covid: pd.DataFrame) -> None:
    section(4, "Aggregation — groupby, agg, pivot_table, nlargest")

    print("4a) Total births per year")
    by_year = baby.groupby("year")["count"].sum().reset_index()
    by_year.columns = ["year", "total_births"]
    print(by_year.tail(5).to_string(index=False))

    print("\n4b) Multiple aggregations — births and unique names per year")
    stats = baby.groupby("year").agg(
        total_births=("count", "sum"),
        unique_names=("name", "nunique"),
        top_count=("count", "max"),
    )
    print(stats.tail(5))

    print("\n4c) groupby + transform — add rank within each (year, sex) group")
    baby2023 = baby.query("year == 2023").copy()
    baby2023["rank"] = baby2023.groupby("sex")["count"].rank(ascending=False, method="min")
    print(baby2023.nsmallest(6, "rank")[["name", "sex", "count", "rank"]].to_string(index=False))

    print("\n4d) pivot_table — top-5 names × decade (total births)")
    top5_names = (
        baby.groupby("name")["count"].sum().nlargest(5).index.tolist()
    )
    baby_top = baby[baby["name"].isin(top5_names)].copy()
    baby_top["decade"] = (baby_top["year"] // 10) * 10
    pt = baby_top.pivot_table(values="count", index="name", columns="decade", aggfunc="sum")
    print(pt.iloc[:, -5:].to_string())   # last 5 decades

    print("""
KEY POINTS
  df.groupby("col")["val"].sum()          → Series; use .reset_index() to get a DF
  df.groupby("col").agg(new=("col","fn")) → named aggregations (pandas ≥ 0.25)
  df.groupby("col")["val"].transform(fn)  → same shape as original (for rank, pct, etc.)
  pd.pivot_table(...)                     → cross-tab; aggfunc can be sum/mean/count
  df.nlargest(n, "col")                   → faster than sort_values + head
""")

    print("""CHALLENGE 4
  Using COVID data: for each location, find the single date with the highest new_cases.
  Return a DataFrame with columns: location, date, new_cases.
  (Hint: groupby location, then use idxmax to get the row index of the max.)
  (Solution at bottom)
""")


# ─────────────────────────────────────────────────────────────
# SECTION 5 — Cleaning
# ─────────────────────────────────────────────────────────────
def s5_cleaning(covid: pd.DataFrame) -> None:
    section(5, "Cleaning — nulls, dtypes, string methods")

    print(f"5a) Null counts in covid:\n{covid.isnull().sum()[covid.isnull().sum() > 0].head(5)}")

    print("\n5b) fillna — fill missing new_cases with 0")
    covid2 = covid.copy()
    covid2["new_cases"] = covid2["new_cases"].fillna(0)
    print(f"Nulls after fill: {covid2['new_cases'].isnull().sum()}")

    print("\n5c) dropna — drop rows missing both new_cases AND new_deaths")
    before = len(covid)
    dropped = covid.dropna(subset=["new_cases", "new_deaths"], how="all")
    print(f"Rows: {before:,} → {len(dropped):,} (dropped {before - len(dropped):,})")

    print("\n5d) String methods — normalize location names")
    locs = pd.Series(["  United States ", "FRANCE", "germany"])
    print(locs.str.strip().str.title())

    print("\n5e) clip and astype")
    covid3 = covid.copy()
    covid3["new_cases"] = covid3["new_cases"].fillna(0).clip(lower=0).astype("int64")
    print(f"new_cases dtype: {covid3['new_cases'].dtype}, min: {covid3['new_cases'].min()}")

    print("""
KEY POINTS
  df.fillna(value)                → fill NaN with a scalar or dict of column→value
  df.fillna(method="ffill")       → forward-fill from previous row
  df.dropna(subset=[...])         → drop rows where specified cols are NaN
  df["col"].str.strip().str.title() → vectorised string operations
  df["col"].clip(lower=0)         → cap values at a floor or ceiling
  df["col"].astype("int64")       → convert dtype
""")

    print("""CHALLENGE 5
  In the COVID DataFrame, create a new column 'case_fatality_rate' =
  (new_deaths / new_cases * 100), rounded to 2 decimal places.
  Set it to NaN where new_cases == 0 to avoid division by zero.
  What is the global median case fatality rate?
  (Solution at bottom)
""")


# ─────────────────────────────────────────────────────────────
# SECTION 6 — Joins and concat
# ─────────────────────────────────────────────────────────────
def s6_joins(baby: pd.DataFrame) -> None:
    section(6, "Joins — pd.merge and pd.concat")

    # Build two mini-DataFrames to demonstrate merge
    rank_2000 = (
        baby.query("year == 2000 and sex == 'F'")
        .nlargest(20, "count")[["name", "count"]]
        .rename(columns={"count": "count_2000"})
        .reset_index(drop=True)
    )
    rank_2023 = (
        baby.query("year == 2023 and sex == 'F'")
        .nlargest(20, "count")[["name", "count"]]
        .rename(columns={"count": "count_2023"})
        .reset_index(drop=True)
    )

    print("6a) INNER JOIN — names in top-20 in BOTH 2000 and 2023")
    inner = pd.merge(rank_2000, rank_2023, on="name", how="inner")
    print(inner.sort_values("count_2023", ascending=False).to_string(index=False))

    print("\n6b) LEFT JOIN — all 2000 top-20; NaN where not in 2023 top-20")
    left = pd.merge(rank_2000, rank_2023, on="name", how="left")
    print(left[left["count_2023"].isna()][["name", "count_2000"]].to_string(index=False))

    print("\n6c) pd.concat — stack two yearly slices")
    slice_a = baby.query("year == 2020")
    slice_b = baby.query("year == 2021")
    combined = pd.concat([slice_a, slice_b], ignore_index=True)
    print(f"2020 rows: {len(slice_a)}, 2021 rows: {len(slice_b)}, combined: {len(combined)}")

    print("""
KEY POINTS
  pd.merge(left, right, on="key", how="inner/left/right/outer")
  how="inner"   → only rows where key exists in both (default)
  how="left"    → all rows from left; NaN where no match in right
  how="outer"   → all rows from both; NaN where no match either side
  pd.concat([df1, df2], axis=0)  → stack rows (axis=1 for columns)
  .merge() can also do multi-key joins: on=["a", "b"]
""")

    print("""CHALLENGE 6
  Find names that were in the top-10 female in 1950 but NOT in the top-100 female in 2023.
  How dramatically did their counts change?
  (Solution at bottom)
""")


# ─────────────────────────────────────────────────────────────
# SECTION 7 — Window functions and time series
# ─────────────────────────────────────────────────────────────
def s7_window(baby: pd.DataFrame, covid: pd.DataFrame) -> None:
    section(7, "Window functions — rolling, rank, pct_change, shift")

    print("7a) 7-day rolling average of new US COVID cases")
    us = (
        covid.query("iso_code == 'USA'")
        .sort_values("date")[["date", "new_cases"]]
        .set_index("date")
    )
    us["rolling_7d"] = us["new_cases"].rolling(7, min_periods=1).mean().round(0)
    print(us.dropna().tail(6))

    print("\n7b) Rank names within each year + sex (window-style rank)")
    recent = baby.query("year >= 2020").copy()
    recent["rank"] = recent.groupby(["year", "sex"])["count"].rank(
        ascending=False, method="min"
    ).astype(int)
    emma_ranks = recent.query("name == 'Emma' and sex == 'F'")[["year", "count", "rank"]]
    print(emma_ranks.to_string(index=False))

    print("\n7c) pct_change — year-over-year birth change per name")
    emma_all = (
        baby.query("name == 'Emma' and sex == 'F'")
        .sort_values("year")
        .set_index("year")[["count"]]
    )
    emma_all["yoy_pct"] = emma_all["count"].pct_change().mul(100).round(1)
    print(emma_all.tail(8))

    print("\n7d) cumsum — running total of births for a name")
    emma_all["cumulative"] = emma_all["count"].cumsum()
    print(f"Emma cumulative births (all years): {emma_all['cumulative'].iloc[-1]:,}")

    print("""
KEY POINTS
  df["col"].rolling(n).mean()          → n-period rolling mean (NaN for first n-1 rows)
  min_periods=1                         → compute even when window is partial
  df.groupby(g)["col"].rank()          → rank within each group (window-style)
  df["col"].pct_change()               → (current - previous) / previous
  df["col"].shift(n)                   → lag by n rows (negative n = lead)
  df["col"].cumsum()                   → cumulative sum
""")

    print("""CHALLENGE 7
  For the US COVID data, calculate a 14-day rolling sum of new_cases.
  Find the 5 dates with the highest 14-day rolling sum.
  (Solution at bottom)
""")


# ─────────────────────────────────────────────────────────────
# SOLUTIONS
# ─────────────────────────────────────────────────────────────
SOLUTIONS = """
╔══════════════════════════════════════════════════════════════╗
║                        SOLUTIONS                             ║
╚══════════════════════════════════════════════════════════════╝

SOLUTION 1 — G7 COVID totals
──────────────────────────────
    G7 = ["USA", "GBR", "DEU", "FRA", "ITA", "JPN", "CAN"]
    g7 = pd.read_csv(COVID_CSV, usecols=["iso_code","location","new_cases"],
                     dtype={"iso_code": "category"})
    g7 = g7[g7["iso_code"].isin(G7)]
    print(g7.groupby("location")["new_cases"].sum().sort_values(ascending=False))

SOLUTION 2 — names in every year
──────────────────────────────────
    all_years = set(range(1880, 2024))
    counts = baby.groupby("name")["year"].nunique()
    persistent = counts[counts == len(all_years)].index.tolist()
    print(len(persistent), persistent[:10])

SOLUTION 3 — high-case locations in 2021
──────────────────────────────────────────
    filtered = covid.query("new_cases > 100_000 and date.dt.year == 2021")
    print(filtered["location"].value_counts().idxmax())

SOLUTION 4 — date of peak new_cases per location
───────────────────────────────────────────────────
    idx = covid.groupby("location")["new_cases"].idxmax()
    peak = covid.loc[idx, ["location", "date", "new_cases"]].reset_index(drop=True)
    print(peak.nlargest(10, "new_cases"))

SOLUTION 5 — case fatality rate
─────────────────────────────────
    c = covid.copy()
    c["cfr"] = (c["new_deaths"] / c["new_cases"].replace(0, float("nan"))) * 100
    c["cfr"] = c["cfr"].round(2)
    print(f"Global median CFR: {c['cfr'].median():.2f}%")

SOLUTION 6 — faded top names
──────────────────────────────
    top10_1950 = baby.query("year==1950 and sex=='F'").nlargest(10,"count")[["name","count"]]
    top100_2023 = baby.query("year==2023 and sex=='F'").nlargest(100,"count")["name"]
    faded = top10_1950[~top10_1950["name"].isin(top100_2023)]
    faded2023 = baby.query("year==2023 and sex=='F'").set_index("name")["count"]
    faded["count_2023"] = faded["name"].map(faded2023).fillna(0).astype(int)
    print(faded.rename(columns={"count":"count_1950"}).to_string(index=False))

SOLUTION 7 — 14-day rolling sum
─────────────────────────────────
    us = covid.query("iso_code=='USA'").sort_values("date").set_index("date")
    us["roll14"] = us["new_cases"].rolling(14).sum()
    print(us["roll14"].nlargest(5))
"""


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
SECTION_FNS = {
    1: lambda b, c: s1_loading(),
    2: s2_exploration,
    3: s3_selection,
    4: s4_aggregation,
    5: lambda b, c: s5_cleaning(c),
    6: lambda b, c: s6_joins(b),
    7: s7_window,
}


def main() -> None:
    _check_data()

    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=int, choices=range(1, 8),
                        help="Run only this section (1-7)")
    parser.add_argument("--solutions", action="store_true")
    args = parser.parse_args()

    if args.solutions:
        print(SOLUTIONS)
        return

    # Always load data (sections share the DataFrames)
    print("Loading datasets …")
    frames = [
        pd.read_csv(f, header=None, names=["name", "sex", "count"]).assign(year=int(f.stem[3:]))
        for f in sorted(BABY_DIR.glob("yob*.txt"))
    ]
    baby = pd.concat(frames, ignore_index=True)
    COLS = ["iso_code", "location", "date", "new_cases", "new_deaths",
            "total_cases", "total_deaths", "population"]
    covid = pd.read_csv(COVID_CSV, usecols=COLS, parse_dates=["date"],
                        dtype={"iso_code": "category", "location": "category"})
    print(f"baby: {baby.shape}  covid: {covid.shape}\n")

    sections_to_run = [args.section] if args.section else list(range(1, 8))
    for n in sections_to_run:
        fn = SECTION_FNS[n]
        if n == 1:
            baby, covid = s1_loading()  # section 1 reloads for demo clarity
        else:
            fn(baby, covid)

    print("\n" + "=" * 62)
    print("  Run with --solutions to see all challenge answers.")
    print("=" * 62)


if __name__ == "__main__":
    main()
