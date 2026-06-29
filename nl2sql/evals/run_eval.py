"""NL2SQL evaluation runner — Phase 1 & 2.

Phase 1 — Gold-only mode (no LLM, no services):
    Runs every gold_sql from gold.jsonl against the DuckDB fixture and verifies
    it executes without error. Proves the gold dataset is internally consistent.

    uv run --extra nl2sql python evals/run_eval.py --gold-only

Phase 2 — Full mode (requires Ollama or OpenAI configured in .env):
    Calls ConversationManager.run_query() for each question and measures
    Execution Accuracy, Exact Match, and secondary metrics from QueryResult.

    uv run --extra nl2sql python evals/run_eval.py
    uv run --extra nl2sql python evals/run_eval.py --model llama3.1:70b

Output:
    evals/results/latest.md          — always overwritten
    evals/results/YYYY-MM-DD_HH.md   — timestamped archive
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import duckdb

_HERE = Path(__file__).resolve().parent   # nl2sql/evals/
_ROOT = _HERE.parent                       # nl2sql/
for _p in (_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

GOLD_PATH = _HERE / "gold.jsonl"
RESULTS_DIR = _HERE / "results"


# ---------------------------------------------------------------------------
# Gold dataset loader
# ---------------------------------------------------------------------------

def load_gold(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _normalize_sql(sql: str) -> str:
    sql = sql.lower().strip().rstrip(";")
    return re.sub(r"\s+", " ", sql)


def exact_match(generated: str, gold: str) -> bool:
    return _normalize_sql(generated) == _normalize_sql(gold)


def execution_match(generated_rows: list[tuple], gold_rows: list[tuple]) -> bool:
    """Order-insensitive comparison — handles equivalent queries that sort differently."""
    try:
        return sorted(str(r) for r in generated_rows) == sorted(str(r) for r in gold_rows)
    except Exception:
        return False


def _run_sql(conn: "duckdb.DuckDBPyConnection", sql: str) -> tuple[list[tuple], str | None]:
    try:
        rows = conn.execute(sql).fetchall()
        return rows, None
    except Exception as exc:
        return [], str(exc)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GoldResult:
    row: dict[str, Any]
    gold_rows: list[tuple] | None      # None = write-guard (no gold SQL)
    gold_error: str | None             # set when gold SQL itself fails
    generated_sql: str | None = None
    generated_rows: list[tuple] | None = None
    generated_error: str | None = None
    attempts: int = 0
    cached: bool = False
    ea: bool | None = None             # None = not applicable (write-guard)
    em: bool | None = None
    guardrail_fired: bool | None = None


# ---------------------------------------------------------------------------
# Phase 1 — Gold-only mode
# ---------------------------------------------------------------------------

def run_gold_only(
    conn: "duckdb.DuckDBPyConnection",
    gold_rows: list[dict[str, Any]],
) -> list[GoldResult]:
    results: list[GoldResult] = []
    for row in gold_rows:
        gold_sql = row.get("gold_sql")

        if gold_sql is None:
            # Write-guard case: no SQL to execute in gold-only mode
            results.append(GoldResult(row=row, gold_rows=None, gold_error=None))
            continue

        rows, error = _run_sql(conn, gold_sql)
        results.append(GoldResult(row=row, gold_rows=rows, gold_error=error))

    return results


# ---------------------------------------------------------------------------
# Phase 2 — Full LLM mode
# ---------------------------------------------------------------------------

async def run_full(
    conn: "duckdb.DuckDBPyConnection",
    gold_rows: list[dict[str, Any]],
    agent: Any,
    schema_text: str,
) -> list[GoldResult]:
    from nlp_sql_postgres_v2 import ConversationManager

    results: list[GoldResult] = []

    for row in gold_rows:
        gold_sql: str | None = row.get("gold_sql")
        expected_error: str | None = row.get("expected_error")

        # Pre-execute gold SQL to get the expected result set
        gold_rows_data: list[tuple] | None = None
        if gold_sql is not None:
            gold_rows_data, _ = _run_sql(conn, gold_sql)

        # Fresh ConversationManager per row — no history leakage between questions
        manager = ConversationManager(
            conn=conn,
            agent=agent,
            schema_text=schema_text,
            max_retries=3,
        )

        try:
            qr = await manager.run_query(row["question"])
        except Exception as exc:
            results.append(GoldResult(
                row=row,
                gold_rows=gold_rows_data,
                gold_error=None,
                generated_error=str(exc),
                attempts=1,
            ))
            continue

        # Write-guard rows: pass if the readonly guardrail fired
        if expected_error == "readonly":
            fired = qr.error is not None and "Only SELECT" in (qr.error or "")
            results.append(GoldResult(
                row=row,
                gold_rows=None,
                gold_error=None,
                generated_sql=qr.sql,
                generated_error=qr.error,
                attempts=qr.attempts,
                cached=qr.cached,
                guardrail_fired=fired,
            ))
            continue

        # Normal query: EA uses rows already in qr (no second execution needed)
        ea = execution_match(list(qr.rows), gold_rows_data or []) if qr.success else False
        em = exact_match(qr.sql, gold_sql) if gold_sql else None

        results.append(GoldResult(
            row=row,
            gold_rows=gold_rows_data,
            gold_error=None,
            generated_sql=qr.sql,
            generated_rows=list(qr.rows) if qr.success else None,
            generated_error=qr.error if not qr.success else None,
            attempts=qr.attempts,
            cached=qr.cached,
            ea=ea,
            em=em,
        ))

    return results


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

_PASS = "✅"
_FAIL = "❌"
_NA   = "—"


def _icon(val: bool | None) -> str:
    if val is None:
        return _NA
    return _PASS if val else _FAIL


def build_report(results: list[GoldResult], mode: str, run_at: datetime) -> str:
    lines: list[str] = []
    lines += [
        "# NL2SQL Evaluation Report",
        "",
        f"**Date:** {run_at.strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Mode:** `{mode}`",
        f"**Gold rows:** {len(results)}",
        "",
    ]

    normal   = [r for r in results if r.row.get("gold_sql") is not None]
    guards   = [r for r in results if r.row.get("expected_error") == "readonly"]

    # ── Summary ──────────────────────────────────────────────────────────────
    lines.append("## Summary")
    lines.append("")

    if mode == "gold-only":
        passed = [r for r in normal if r.gold_error is None]
        failed = [r for r in normal if r.gold_error is not None]
        lines.append(
            f"Gold SQL sanity check: **{len(passed)}/{len(normal)} passed** "
            f"({'all good' if not failed else f'{len(failed)} failed'})"
        )
        if failed:
            lines.append("")
            for r in failed:
                lines.append(f"- `{r.row['id']}` — {r.gold_error}")
    else:
        ea_pass  = sum(1 for r in normal if r.ea)
        em_pass  = sum(1 for r in normal if r.em)
        retried  = [r for r in normal if r.attempts and r.attempts > 1]
        gd_pass  = sum(1 for r in guards if r.guardrail_fired)
        valid    = sum(1 for r in normal if not r.generated_error)
        n        = len(normal)

        ea_pct   = ea_pass / n if n else 0.0
        em_pct   = em_pass / n if n else 0.0
        vld_pct  = valid   / n if n else 0.0
        ret_pct  = len(retried) / n if n else 0.0
        mean_att = sum(r.attempts for r in normal if r.attempts) / n if n else 0.0
        gd_pct   = gd_pass / len(guards) if guards else 0.0

        lines += [
            "| Metric | Score | Target | Status |",
            "|--------|-------|--------|--------|",
            f"| Execution Accuracy | {ea_pct:.1%} ({ea_pass}/{n}) | ≥ 80% | {_PASS if ea_pct >= 0.80 else _FAIL} |",
            f"| Exact Match        | {em_pct:.1%} ({em_pass}/{n}) | ≥ 55% | {_PASS if em_pct >= 0.55 else _FAIL} |",
            f"| Valid SQL Rate     | {vld_pct:.1%} | ≥ 95% | {_PASS if vld_pct >= 0.95 else _FAIL} |",
            f"| Retry Rate         | {ret_pct:.1%} | — | {_NA} |",
            f"| Mean Attempts      | {mean_att:.2f} | ≤ 1.30 | {_PASS if mean_att <= 1.30 else _FAIL} |",
            f"| Write-Guard Rate   | {gd_pct:.1%} ({gd_pass}/{len(guards)}) | 100% | {_PASS if gd_pct == 1.0 else _FAIL} |",
        ]

    lines += ["", "---", "", "## Per-Query Results", ""]

    # ── Per-query ─────────────────────────────────────────────────────────────
    for r in results:
        row = r.row
        is_guard = row.get("expected_error") == "readonly"
        tags_str = " ".join(f"`{t}`" for t in row.get("tags", []))

        lines.append(f"### {row['id']}: {row['question']}")
        lines.append("")
        lines.append(f"**Difficulty:** `{row['difficulty']}`  |  **Tags:** {tags_str}")
        lines.append("")

        if mode == "gold-only":
            if is_guard:
                lines.append("_Write-guard case — guardrail tested in full mode only_")
            elif r.gold_error:
                lines.append(f"{_FAIL} **FAIL** — `{r.gold_error}`")
                lines.append(f"```sql\n{row['gold_sql']}\n```")
            else:
                row_count = len(r.gold_rows) if r.gold_rows is not None else 0
                lines.append(f"{_PASS} **PASS** — {row_count} row(s) returned")
                lines.append(f"```sql\n{row['gold_sql']}\n```")
        else:
            if is_guard:
                lines.append(f"**Write-guard:** {_icon(r.guardrail_fired)} {'fired' if r.guardrail_fired else 'did NOT fire'}")
                if r.generated_sql:
                    lines.append(f"```sql\n{r.generated_sql}\n```")
            else:
                lines.append(
                    f"**EA:** {_icon(r.ea)}  "
                    f"**EM:** {_icon(r.em)}  "
                    f"**Attempts:** {r.attempts}  "
                    f"**Cached:** {r.cached}"
                )
                if r.generated_sql:
                    lines.append(f"```sql\n{r.generated_sql}\n```")
                if r.generated_error:
                    lines.append(f"> **Error:** {r.generated_error}")

        lines += ["", "---", ""]

    # ── Raw JSON ──────────────────────────────────────────────────────────────
    lines += ["## Raw Results (JSON)", "", "```json"]
    raw: list[dict[str, Any]] = []
    for r in results:
        entry: dict[str, Any] = {
            "id":         r.row["id"],
            "question":   r.row["question"],
            "difficulty": r.row["difficulty"],
            "tags":       r.row.get("tags", []),
        }
        if mode == "gold-only":
            entry["gold_sql_ok"] = r.gold_error is None and r.row.get("gold_sql") is not None
            entry["gold_error"]  = r.gold_error
        else:
            entry["ea"]              = r.ea
            entry["em"]              = r.em
            entry["attempts"]        = r.attempts
            entry["guardrail_fired"] = r.guardrail_fired
            entry["generated_sql"]   = r.generated_sql
            entry["generated_error"] = r.generated_error
        raw.append(entry)
    lines += [json.dumps(raw, indent=2), "```", ""]

    return "\n".join(lines)


def write_report(report: str, output_dir: Path, run_at: datetime) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest  = output_dir / "latest.md"
    archive = output_dir / f"{run_at.strftime('%Y-%m-%d_%H%M%S')}.md"
    latest.write_text(report, encoding="utf-8")
    archive.write_text(report, encoding="utf-8")
    return latest, archive


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    from fixtures import build_sales_fixture, SCHEMA_TEXT

    gold_rows = load_gold(Path(args.gold_path))
    conn = build_sales_fixture()
    run_at = datetime.now(UTC)

    if args.gold_only:
        results = run_gold_only(conn, gold_rows)
        mode = "gold-only"
    else:
        from nlp_sql_postgres_v2 import load_env
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIModel

        load_env()
        model = OpenAIModel(
            model_name=args.model,
            base_url=args.base_url,
            api_key=args.api_key or "ollama",
        )
        agent: Agent[None, str] = Agent(model=model, output_type=str)
        results = await run_full(conn, gold_rows, agent, SCHEMA_TEXT)
        mode = "full"

    report = build_report(results, mode, run_at)
    latest, archive = write_report(report, RESULTS_DIR, run_at)

    print(f"Report  → {latest}")
    print(f"Archive → {archive}")

    # Exit non-zero on failure so CI can gate on it
    if args.gold_only:
        failed = [r for r in results if r.gold_error]
        if failed:
            print(f"\n{len(failed)} gold SQL(s) failed — fix gold.jsonl before running full eval",
                  file=sys.stderr)
            sys.exit(1)
    else:
        normal = [r for r in results if r.row.get("gold_sql") is not None]
        ea = sum(1 for r in normal if r.ea) / len(normal) if normal else 0.0
        if ea < 0.80:
            print(f"\nEA {ea:.1%} is below the 80% target — regression detected", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NL2SQL evaluation runner")
    parser.add_argument(
        "--gold-only", action="store_true",
        help="Sanity-check gold SQL against the DuckDB fixture only (no LLM)",
    )
    parser.add_argument(
        "--gold-path", default=str(GOLD_PATH),
        help="Path to gold.jsonl (default: evals/gold.jsonl)",
    )
    parser.add_argument(
        "--model", default="llama3.2:3b",
        help="LLM model name for full mode (default: llama3.2:3b)",
    )
    parser.add_argument(
        "--base-url", default="http://localhost:11434/v1",
        help="LLM base URL for full mode (default: Ollama local)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key for full mode (default: 'ollama')",
    )
    asyncio.run(main(parser.parse_args()))
