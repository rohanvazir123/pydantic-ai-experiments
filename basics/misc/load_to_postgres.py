"""
Loads two public-domain datasets into local PostgreSQL.

Tables created
──────────────
  baby_names(year, name, sex, count)
      Source: US Baby Names ZIP (SSA) — all years from 1880

  world_gdp(country_code, country_name, year, gdp_usd)
      Source: World Bank GDP (current USD) XLS

Both tables are dropped and recreated on each run (idempotent).
For the XLS load, requires:  pip install xlrd

Usage:
    python load_to_postgres.py                 # load both tables
    python load_to_postgres.py --table baby    # load only baby_names
    python load_to_postgres.py --table gdp     # load only world_gdp

Environment variables (see coding_fun/postgres/config.py pattern):
    PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import asyncpg

try:
    import pandas as pd
except ImportError:
    sys.exit("Install pandas: pip install pandas")

DATASETS = Path(__file__).parent / "datasets"
BABY_DIR = DATASETS / "baby_names"
GDP_DIR = DATASETS / "world_bank_gdp"


def _load_dotenv() -> None:
    """Load a .env file from this directory without requiring python-dotenv."""
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# ── connection ────────────────────────────────────────────────
HOST = os.getenv("PG_HOST", "localhost")
PORT = int(os.getenv("PG_PORT", "5432"))
USER = os.getenv("PG_USER", "postgres")
PASSWORD = os.getenv("PG_PASSWORD", "")
DATABASE = os.getenv("PG_DATABASE", "postgres")


def _dsn() -> str:
    if PASSWORD:
        return f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
    return f"postgresql://{USER}@{HOST}:{PORT}/{DATABASE}"


# ─────────────────────────────────────────────────────────────
# baby_names
# ─────────────────────────────────────────────────────────────
CREATE_BABY = """
DROP TABLE IF EXISTS baby_names;
CREATE TABLE baby_names (
    year    SMALLINT NOT NULL,
    name    TEXT     NOT NULL,
    sex     CHAR(1)  NOT NULL,
    count   INTEGER  NOT NULL
);
CREATE INDEX idx_baby_year  ON baby_names(year);
CREATE INDEX idx_baby_name  ON baby_names(name);
CREATE INDEX idx_baby_sex   ON baby_names(sex);
"""


def _load_baby_df() -> pd.DataFrame:
    if not BABY_DIR.exists():
        sys.exit("datasets/baby_names/ not found — run download.py first.")
    frames = [
        pd.read_csv(f, header=None, names=["name", "sex", "count"]).assign(year=int(f.stem[3:]))
        for f in sorted(BABY_DIR.glob("yob*.txt"))
    ]
    df = pd.concat(frames, ignore_index=True)[["year", "name", "sex", "count"]]
    return df


async def load_baby(conn: asyncpg.Connection) -> None:
    print("Loading baby_names …")
    t0 = time.perf_counter()

    df = _load_baby_df()
    print(f"  Rows to insert: {len(df):,}  ({df['year'].min()}–{df['year'].max()})")

    await conn.execute(CREATE_BABY)

    rows = list(df.itertuples(index=False, name=None))
    await conn.executemany(
        "INSERT INTO baby_names(year, name, sex, count) VALUES($1, $2, $3, $4)",
        rows,
    )

    actual = await conn.fetchval("SELECT count(*) FROM baby_names")
    elapsed = time.perf_counter() - t0
    print(f"  Done — {actual:,} rows in {elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────
# world_gdp
# ─────────────────────────────────────────────────────────────
CREATE_GDP = """
DROP TABLE IF EXISTS world_gdp;
CREATE TABLE world_gdp (
    country_code  CHAR(3)          NOT NULL,
    country_name  TEXT             NOT NULL,
    year          SMALLINT         NOT NULL,
    gdp_usd       DOUBLE PRECISION
);
CREATE INDEX idx_gdp_code ON world_gdp(country_code);
CREATE INDEX idx_gdp_year ON world_gdp(year);
"""


def _find_xls() -> Path:
    """Locate the World Bank XLS inside datasets/world_bank_gdp/."""
    if not GDP_DIR.exists():
        sys.exit("datasets/world_bank_gdp/ not found — run download.py first.")
    candidates = list(GDP_DIR.glob("API_NY.GDP*.xls")) + list(GDP_DIR.glob("API_NY.GDP*.xlsx"))
    if not candidates:
        sys.exit(
            "No World Bank XLS file found in datasets/world_bank_gdp/.\n"
            "Run download.py to fetch it."
        )
    return candidates[0]


def _load_gdp_df() -> pd.DataFrame:
    xls_path = _find_xls()
    print(f"  Reading {xls_path.name} …")

    try:
        # World Bank XLS files are old-format BIFF8 — requires xlrd
        raw = pd.read_excel(xls_path, engine="xlrd", header=3, skipfooter=1)
    except ImportError:
        sys.exit("xlrd is required to read .xls files:\n  pip install xlrd")
    except Exception:
        # Fallback: try openpyxl (some World Bank downloads are XLSX despite .xls extension)
        try:
            raw = pd.read_excel(xls_path, engine="openpyxl", header=3, skipfooter=1)
        except Exception as exc:
            sys.exit(f"Could not read {xls_path.name}: {exc}\n  pip install xlrd openpyxl")

    # Columns: Country Name, Country Code, Indicator Name, Indicator Code, 1960, 1961, …
    year_cols = [c for c in raw.columns if str(c).isdigit()]
    id_cols = {"Country Name": "country_name", "Country Code": "country_code"}
    df = raw.rename(columns=id_cols)[["country_name", "country_code"] + year_cols]

    # Melt: wide → long
    df = df.melt(
        id_vars=["country_name", "country_code"],
        value_vars=year_cols,
        var_name="year",
        value_name="gdp_usd",
    )
    df["year"] = df["year"].astype(int)
    df = df[df["country_code"].notna() & (df["country_code"].str.len() == 3)]
    df = df[["country_code", "country_name", "year", "gdp_usd"]].reset_index(drop=True)
    return df


async def load_gdp(conn: asyncpg.Connection) -> None:
    print("Loading world_gdp …")
    t0 = time.perf_counter()

    df = _load_gdp_df()
    print(f"  Rows to insert: {len(df):,}")

    await conn.execute(CREATE_GDP)

    # Replace NaN with None so asyncpg inserts NULL
    df = df.where(pd.notna(df), other=None)
    rows = list(df.itertuples(index=False, name=None))
    await conn.executemany(
        "INSERT INTO world_gdp(country_code, country_name, year, gdp_usd) VALUES($1,$2,$3,$4)",
        rows,
    )

    actual = await conn.fetchval("SELECT count(*) FROM world_gdp")
    elapsed = time.perf_counter() - t0
    print(f"  Done — {actual:,} rows in {elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
async def main(table: str | None) -> None:
    print(f"Connecting to {HOST}:{PORT}/{DATABASE} as {USER}\n")
    conn: asyncpg.Connection = await asyncpg.connect(_dsn())

    try:
        if table in (None, "baby"):
            await load_baby(conn)
            print()
        if table in (None, "gdp"):
            await load_gdp(conn)
            print()

        # Quick sanity check
        print("Row counts:")
        for tbl in ("baby_names", "world_gdp"):
            try:
                n = await conn.fetchval(f"SELECT count(*) FROM {tbl}")
                print(f"  {tbl:20s} {n:>10,}")
            except asyncpg.UndefinedTableError:
                print(f"  {tbl:20s}  (not loaded)")
    finally:
        await conn.close()

    print("\nReady for sql_challenges.sql")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table", choices=["baby", "gdp"],
        help="Load only this table (default: both)"
    )
    asyncio.run(main(parser.parse_args().table))
