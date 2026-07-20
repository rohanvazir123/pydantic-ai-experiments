"""Download a single database from the BIRD Mini-Dev benchmark.

BIRD Mini-Dev (https://github.com/bird-bench/mini_dev) ships as one ~800MB
zip containing every database. Rather than pulling the whole archive, this
uses HTTP range requests (via `remotezip`) to fetch only the requested
database's files, plus the matching subset of eval artifacts:

  - <db_id>.sqlite + database_description/*.csv  (raw tables)
  - schema.sql             CREATE TABLE/INDEX statements, dumped via sqlite3 CLI
  - schema.json            spider-format column/table metadata (dev_tables.json)
  - questions.json         NL questions for this db (mini_dev_sqlite.json)
  - gold.sql               matching gold SQL, one per line (mini_dev_sqlite_gold.sql)

Usage:
    uv run --with remotezip python scripts/download_bird_mini_dev.py
    uv run --with remotezip python scripts/download_bird_mini_dev.py --db-id card_games
    uv run --with remotezip python scripts/download_bird_mini_dev.py --list-dbs
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from remotezip import RemoteZip

MINIDEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
ZIP_ROOT = "minidev/MINIDEV"


def list_db_ids(z: RemoteZip) -> list[str]:
    prefix = f"{ZIP_ROOT}/dev_databases/"
    seen: list[str] = []
    for name in z.namelist():
        if name.startswith(prefix) and name.endswith("/") and name.count("/") == prefix.count("/") + 1:
            db_id = name[len(prefix) :].rstrip("/")
            if db_id:
                seen.append(db_id)
    return sorted(seen)


def download_db(z: RemoteZip, db_id: str, dest_dir: Path) -> None:
    prefix = f"{ZIP_ROOT}/dev_databases/{db_id}/"
    matches = [n for n in z.namelist() if n.startswith(prefix) and not n.endswith("/")]
    if not matches:
        raise SystemExit(f"No files found for db_id={db_id!r} under {prefix}")

    for src in matches:
        rel = src[len(prefix) :]
        out_path = dest_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with z.open(src) as fin, open(out_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        print(f"wrote {out_path}")


def dump_schema_sql(sqlite_path: Path, dest_dir: Path) -> None:
    """Dump CREATE TABLE/INDEX statements from the sqlite db via the sqlite3 CLI."""
    out_path = dest_dir / "schema.sql"
    result = subprocess.run(
        ["sqlite3", str(sqlite_path), ".schema"],
        capture_output=True,
        text=True,
        check=True,
    )
    out_path.write_text(result.stdout)
    print(f"wrote {out_path}")


def filter_questions(z: RemoteZip, db_id: str, dest_dir: Path) -> None:
    with z.open(f"{ZIP_ROOT}/mini_dev_sqlite.json") as f:
        questions = json.load(f)
    with z.open(f"{ZIP_ROOT}/mini_dev_sqlite_gold.sql") as f:
        gold_lines = f.read().decode("utf-8").splitlines()

    assert len(questions) == len(gold_lines), "questions/gold line count mismatch"

    filtered_questions = []
    filtered_gold = []
    for q, gold_line in zip(questions, gold_lines):
        if q["db_id"] == db_id:
            filtered_questions.append(q)
            filtered_gold.append(gold_line)

    if not filtered_questions:
        print(f"warning: no questions found for db_id={db_id!r}")

    (dest_dir / "questions.json").write_text(json.dumps(filtered_questions, indent=2))
    (dest_dir / "gold.sql").write_text("\n".join(filtered_gold) + "\n")
    print(f"wrote {dest_dir / 'questions.json'} ({len(filtered_questions)} questions)")
    print(f"wrote {dest_dir / 'gold.sql'}")


def filter_schema(z: RemoteZip, db_id: str, dest_dir: Path) -> None:
    with z.open(f"{ZIP_ROOT}/dev_tables.json") as f:
        tables = json.load(f)
    match = next((t for t in tables if t["db_id"] == db_id), None)
    if match is None:
        print(f"warning: no schema entry found for db_id={db_id!r}")
        return
    out_path = dest_dir / "schema.json"
    out_path.write_text(json.dumps(match, indent=2))
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-id", default="california_schools", help="Database to download (default: california_schools)")
    parser.add_argument(
        "--dest",
        default=None,
        help="Output directory (default: data/bird_mini_dev/<db-id> relative to this script's parent)",
    )
    parser.add_argument("--list-dbs", action="store_true", help="List available db_ids in Mini-Dev and exit")
    args = parser.parse_args()

    with RemoteZip(MINIDEV_URL) as z:
        if args.list_dbs:
            for db_id in list_db_ids(z):
                print(db_id)
            return

        dest_dir = Path(args.dest) if args.dest else Path(__file__).parent.parent / "data" / "bird_mini_dev" / args.db_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        download_db(z, args.db_id, dest_dir)
        dump_schema_sql(dest_dir / f"{args.db_id}.sqlite", dest_dir)
        filter_questions(z, args.db_id, dest_dir)
        filter_schema(z, args.db_id, dest_dir)


if __name__ == "__main__":
    main()
