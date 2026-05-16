"""
Retrieval evaluation pipeline — measures KG query quality.

Metrics computed per question
------------------------------
- intent_matched     : IntentParser produced a non-fallback intent
- cypher_valid       : Cypher contains required structural keywords (MATCH + RETURN + LIMIT)
- result_non_empty   : AGE returned at least one row
- latency_ms         : end-to-end wall time (intent → AGE response)

Aggregate
---------
- intent_match_rate  : fraction of queries with a specific intent
- result_hit_rate    : fraction of queries that returned ≥1 row
- mean_latency_ms
- p95_latency_ms

Usage
-----
    # Evaluate all built-in test queries (no AGE needed — dry-run mode)
    python -m kg.legal.retrieval.eval_pipeline --dry-run

    # Full evaluation against live AGE (requires docker-compose up -d)
    python -m kg.legal.retrieval.eval_pipeline

    # Save results to JSON
    python -m kg.legal.retrieval.eval_pipeline --output kg/evals/retrieval_eval_latest.json

    # Evaluate a custom question file (one question per line)
    python -m kg.legal.retrieval.eval_pipeline --questions my_questions.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, quantiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

from kg.legal.retrieval.intent_parser import IntentParser
from kg.legal.retrieval.nl2cypher import NL2CypherConverter

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Built-in evaluation questions
# (sampled from the known intent space — one per intent category)
# ---------------------------------------------------------------------------

EVAL_QUESTIONS: list[tuple[str, str]] = [
    # (expected_intent_prefix, question)
    ("find_parties",                "Who are the parties to the Strategic Alliance Agreement?"),
    ("find_parties",                "Who are the parties?"),
    ("find_indemnification",        "Which parties indemnify each other?"),
    ("find_indemnification",        "Who indemnifies whom in the Lightbridge contract?"),
    ("find_jurisdictions",          "What is the governing law?"),
    ("find_jurisdictions",          "What jurisdiction governs the Strategic Alliance Agreement?"),
    ("find_termination_clauses",    "What are the termination clauses?"),
    ("find_termination_clauses",    "What does the termination clause say in the Lightbridge contract?"),
    ("find_confidentiality_clauses","Are there any confidentiality obligations?"),
    ("find_confidentiality_clauses","What are the NDA terms in the Strategic Alliance Agreement?"),
    ("find_payment_terms",          "What payment terms exist?"),
    ("find_payment_terms",          "What are the fees in the Lightbridge agreement?"),
    ("find_obligations",            "What obligations does the contract impose?"),
    ("find_liability_clauses",      "What is the limitation of liability?"),
    ("find_effective_dates",        "What is the effective date?"),
    ("find_effective_dates",        "When does the Strategic Alliance Agreement take effect?"),
    ("find_expiration_dates",       "When does the contract expire?"),
    ("find_renewal_terms",          "What are the renewal terms?"),
    ("find_renewal_terms",          "Does the contract auto-renew?"),
    ("find_disclosures",            "What disclosures are made between parties?"),
    ("find_sections",               "What sections does the contract have?"),
    ("find_sections",               "What sections are in the Strategic Alliance Agreement?"),
    ("find_superseded_contracts",   "Which contracts supersede others?"),
    ("find_amendments",             "Which contracts have been amended?"),
    ("find_references",             "What documents are referenced?"),
    ("find_incorporated_documents", "What documents does the contract incorporate by reference?"),
    ("find_attachments",            "What is attached to the Strategic Alliance Agreement?"),
    ("find_replacements",           "Which contracts replace older agreements?"),
    ("find_all_risks",              "What are the compliance risks?"),
    ("find_all_risks",              "What risks affect Lightbridge?"),
    ("find_risk_chains",            "What risk factors cause other risks?"),
    ("find_missing_indemnity",      "Which contracts lack an indemnity clause?"),
    ("find_missing_termination",    "Which contracts are missing a termination clause?"),
    ("list_contracts",              "List all contracts."),
]

_FALLBACK_INTENT = "list_contracts"


@dataclass
class QueryResult:
    question: str
    expected_intent: str
    actual_intent: str
    params: dict
    cypher: str
    result_rows: int
    latency_ms: float
    error: str = ""
    intent_matched: bool = False
    cypher_valid: bool = False
    result_non_empty: bool = False


@dataclass
class EvalReport:
    total: int = 0
    intent_match_rate: float = 0.0
    result_hit_rate: float = 0.0
    cypher_valid_rate: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    results: list[QueryResult] = field(default_factory=list)


def _cypher_is_valid(cypher: str) -> bool:
    upper = cypher.upper()
    return "MATCH" in upper and "RETURN" in upper and "LIMIT" in upper


async def _evaluate_one(
    question: str,
    expected_intent: str,
    converter: NL2CypherConverter,
    parser: IntentParser,
    store,
    dry_run: bool,
) -> QueryResult:
    t0 = time.perf_counter()
    match = parser.parse(question)
    cypher = await converter.convert(question)

    result_rows = 0
    error = ""

    if not dry_run and store is not None:
        try:
            raw = await store.run_cypher_query(cypher)
            lines = [ln for ln in raw.splitlines() if ln.strip()]
            result_rows = max(0, len(lines) - 1)  # subtract header row
        except Exception as exc:
            error = str(exc)
            logger.warning("Query error for %r: %s", question, exc)

    latency_ms = (time.perf_counter() - t0) * 1000

    return QueryResult(
        question=question,
        expected_intent=expected_intent,
        actual_intent=match.intent,
        params=match.params,
        cypher=cypher,
        result_rows=result_rows,
        latency_ms=latency_ms,
        error=error,
        intent_matched=(match.intent == expected_intent),
        cypher_valid=_cypher_is_valid(cypher),
        result_non_empty=(result_rows > 0),
    )


async def run_eval(
    questions: list[tuple[str, str]],
    dry_run: bool = False,
) -> EvalReport:
    converter = NL2CypherConverter()
    parser = IntentParser()
    store = None

    if not dry_run:
        from kg.age_graph_store import AgeGraphStore
        store = AgeGraphStore()
        await store.initialize()

    try:
        tasks = [
            _evaluate_one(q, intent, converter, parser, store, dry_run)
            for intent, q in questions
        ]
        results = await asyncio.gather(*tasks)
    finally:
        if store is not None:
            await store.close()

    n = len(results)
    latencies = [r.latency_ms for r in results]
    report = EvalReport(
        total=n,
        intent_match_rate=sum(r.intent_matched for r in results) / n,
        result_hit_rate=sum(r.result_non_empty for r in results) / n,
        cypher_valid_rate=sum(r.cypher_valid for r in results) / n,
        mean_latency_ms=mean(latencies),
        p95_latency_ms=quantiles(latencies, n=20)[18] if n >= 20 else max(latencies),
        results=list(results),
    )
    return report


def _print_report(report: EvalReport, show_all: bool = False) -> None:
    console.print(f"\n[bold]Retrieval Eval — {report.total} questions[/bold]")

    summary = Table(show_header=True, header_style="bold cyan")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Intent match rate",  f"{report.intent_match_rate:.1%}")
    summary.add_row("Cypher valid rate",  f"{report.cypher_valid_rate:.1%}")
    if report.result_hit_rate > 0:
        summary.add_row("Result hit rate",   f"{report.result_hit_rate:.1%}")
    summary.add_row("Mean latency",       f"{report.mean_latency_ms:.1f} ms")
    summary.add_row("P95 latency",        f"{report.p95_latency_ms:.1f} ms")
    console.print(summary)

    failures = [r for r in report.results if not r.intent_matched or not r.cypher_valid]
    if failures:
        console.print(f"\n[yellow]Intent/Cypher failures ({len(failures)}):[/yellow]")
        fail_table = Table(show_header=True, header_style="bold yellow")
        fail_table.add_column("Question", max_width=50)
        fail_table.add_column("Expected")
        fail_table.add_column("Got")
        fail_table.add_column("Cypher OK")
        for r in failures:
            fail_table.add_row(
                r.question[:50],
                r.expected_intent,
                r.actual_intent,
                "OK" if r.cypher_valid else "NO",
            )
        console.print(fail_table)

    if show_all:
        console.print("\n[bold]All results:[/bold]")
        all_table = Table(show_header=True, header_style="bold")
        all_table.add_column("Question", max_width=45)
        all_table.add_column("Intent", max_width=28)
        all_table.add_column("Rows", justify="right")
        all_table.add_column("ms", justify="right")
        all_table.add_column("OK")
        for r in report.results:
            ok = "OK" if r.intent_matched and r.cypher_valid else "NO"
            all_table.add_row(
                r.question[:45],
                r.actual_intent,
                str(r.result_rows) if r.result_rows >= 0 else "-",
                f"{r.latency_ms:.0f}",
                ok,
            )
        console.print(all_table)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate KG retrieval quality.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip AGE queries (test intent + Cypher only)")
    ap.add_argument("--output", help="Save JSON report to this path")
    ap.add_argument("--questions", help="Text file with one question per line (no expected intent)")
    ap.add_argument("--show-all", action="store_true",
                    help="Print all results, not just failures")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    questions = list(EVAL_QUESTIONS)
    if args.questions:
        lines = Path(args.questions).read_text().splitlines()
        questions = [("list_contracts", ln.strip()) for ln in lines if ln.strip()]
        console.print(f"[dim]Loaded {len(questions)} questions from {args.questions}[/dim]")

    report = asyncio.run(run_eval(questions, dry_run=args.dry_run))
    _print_report(report, show_all=args.show_all)

    if args.output:
        out = {
            "total": report.total,
            "intent_match_rate": report.intent_match_rate,
            "result_hit_rate": report.result_hit_rate,
            "cypher_valid_rate": report.cypher_valid_rate,
            "mean_latency_ms": report.mean_latency_ms,
            "p95_latency_ms": report.p95_latency_ms,
            "results": [asdict(r) for r in report.results],
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        console.print(f"\n[green]Report saved to {args.output}[/green]")


if __name__ == "__main__":
    main()
