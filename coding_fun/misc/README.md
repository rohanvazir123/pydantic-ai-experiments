# SQL, Pandas & CSV Crash Course

Public-domain datasets + hands-on scripts for brushing up on Python's `csv` module, `pandas`, SQL, and loading Excel files into PostgreSQL.

## Datasets

| Dataset | Source | Format | Size |
|---------|--------|--------|------|
| **US Baby Names** (1880–present) | Social Security Administration | ZIP of ~140 CSV files | ~25 MB extracted |
| **Our World in Data — COVID-19** | OWID | Single CSV, 300k+ rows, 67 columns | ~100 MB |
| **World Bank — GDP (current USD)** | World Bank Open Data API | ZIP → XLS (wide format) | ~3 MB |

All three are in the public domain / licensed for free use.

## Manual Downloads

The SSA blocks programmatic downloads of the baby names ZIP. Download all three files manually and place them in `datasets/` before running `download.py`.

| Dataset | URL | Save as |
|---------|-----|---------|
| **US Baby Names** | Go to https://www.ssa.gov/oact/babynames/limits.html → click **"National data"** | `datasets/baby_names.zip` |
| **OWID COVID CSV** | https://covid.ourworldindata.org/data/owid-covid-data.csv | `datasets/owid_covid.csv` |
| **World Bank GDP** | https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.CD?downloadformat=excel | `datasets/world_bank_gdp.zip` |

Once the files are in place, run `python download.py` — it detects existing files (cached), extracts the ZIPs, and reports everything ready.

## Quick Start

```bash
cd coding_fun/misc

# 1. Install dependencies
pip install pandas openpyxl requests
pip install xlrd   # needed to read the World Bank .xls file

# 2. Extract downloaded datasets (after manual download above)
python download.py

# 3. Work through the crash courses
python csv_crash_course.py        # Python csv module
python pandas_crash_course.py     # pandas

# 4. Load data into local PostgreSQL
python load_to_postgres.py

# 5. Work through SQL challenges
psql -d postgres -f sql_challenges.sql
```

## Files

```
misc/
├── download.py              Download all three datasets to datasets/
├── csv_crash_course.py      5-section csv module tutorial + challenges
├── pandas_crash_course.py   7-section pandas tutorial + challenges
├── load_to_postgres.py      Load baby_names + world_gdp into local PostgreSQL
├── sql_challenges.sql       20 SQL challenges (basics → window functions)
└── datasets/                Downloaded files land here (add to .gitignore)
```

## What Each Script Covers

### `csv_crash_course.py`

| Section | Topic |
|---------|-------|
| 1 | `csv.reader` — raw row iteration, type casting |
| 2 | `csv.DictReader` — named-field access, no-header files |
| 3 | Multi-file reads — tracking a name across 40+ year files |
| 4 | `csv.writer` / `csv.DictWriter` — writing filtered output |
| 5 | `csv.Sniffer` + `io.StringIO` — dialect detection, in-memory CSV |

Each section ends with a challenge. Run `--solutions` to reveal answers:

```bash
python csv_crash_course.py --section 3
python csv_crash_course.py --solutions
```

### `pandas_crash_course.py`

| Section | Topic |
|---------|-------|
| 1 | Loading: `read_csv`, `concat`, `usecols`, `parse_dates`, `dtype` |
| 2 | Exploration: `info`, `describe`, `value_counts`, `isnull` |
| 3 | Selection: `loc`, `iloc`, boolean indexing, `query`, `isin` |
| 4 | Aggregation: `groupby`, `agg`, `transform`, `pivot_table` |
| 5 | Cleaning: `fillna`, `dropna`, `clip`, `astype`, `str.` methods |
| 6 | Joins: `pd.merge` (inner/left/outer), `pd.concat` |
| 7 | Window: `rolling`, `rank`, `pct_change`, `shift`, `cumsum` |

```bash
python pandas_crash_course.py --section 4
python pandas_crash_course.py --solutions
```

### `load_to_postgres.py`

Creates two tables in your local PostgreSQL database:

```sql
baby_names(year SMALLINT, name TEXT, sex CHAR(1), count INTEGER)
world_gdp(country_code CHAR(3), country_name TEXT, year SMALLINT, gdp_usd DOUBLE PRECISION)
```

The XLS → PostgreSQL pipeline:
1. Reads the World Bank XLS with `pandas.read_excel`
2. Melts from wide format (one column per year) to long format
3. Bulk-inserts via `asyncpg.executemany`

```bash
python load_to_postgres.py            # load both tables
python load_to_postgres.py --table baby
python load_to_postgres.py --table gdp
```

### `sql_challenges.sql`

20 progressive challenges — copy-paste into psql or any SQL client.

| Challenge | Concept |
|-----------|---------|
| 01 | `SELECT`, `WHERE`, `ORDER BY`, `LIMIT` |
| 02 | Column aliases, arithmetic, scalar subquery |
| 03 | `BETWEEN`, `IN`, `LIKE` |
| 04 | `DISTINCT`, `COUNT(DISTINCT ...)` |
| 05 | `GROUP BY` + `SUM`, `AVG`, `MAX` |
| 06 | `HAVING` — filter after aggregation |
| 07 | `CASE WHEN` inside aggregation (pivot-style) |
| 08 | Scalar subquery in `WHERE` |
| 09 | CTE (`WITH` clause) |
| 10 | `EXISTS` / `NOT EXISTS` |
| 11 | Self-join |
| 12 | `JOIN` across two tables |
| 13 | `LEFT JOIN` + NULL detection |
| 14 | `ROW_NUMBER() OVER (PARTITION BY ...)` |
| 15 | `RANK` vs `DENSE_RANK` |
| 16 | `LAG()` — year-over-year change |
| 17 | Running total with `SUM() OVER (ORDER BY ...)` |
| 18 | `NTILE(n)` — quartile bucketing |
| 19 | GDP growth rate (LAG + arithmetic + CTE) |
| 20 | Most stable name — `stddev` + `HAVING` on years |

Each challenge shows the task, a hint, and a commented-out solution.

## PostgreSQL Connection

Scripts read the same env vars as `coding_fun/postgres/config.py`:

```bash
export PG_HOST=localhost     # default
export PG_PORT=5432          # default
export PG_USER=postgres      # default
export PG_PASSWORD=          # default (empty = trust auth)
export PG_DATABASE=postgres  # default
```

## .gitignore

Add the datasets folder to avoid committing hundreds of MBs:

```
coding_fun/misc/datasets/
```
