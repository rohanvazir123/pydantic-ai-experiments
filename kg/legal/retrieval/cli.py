"""
Interactive CLI for querying the legal knowledge graph.

Routes each question through the full retrieval pipeline:
  NL question → IntentParser → QUERY_CAPABILITIES → Cypher → AGE → formatted answer

Usage
-----
    # Interactive REPL
    python -m kg.legal.retrieval.cli

    # Single question (non-interactive)
    python -m kg.legal.retrieval.cli --question "Who are the parties to the Lightbridge contract?"

    # Show raw Cypher alongside results
    python -m kg.legal.retrieval.cli --show-cypher

    # Pipe questions from stdin
    echo "What are the termination clauses?" | python -m kg.legal.retrieval.cli --stdin
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kg.age_graph_store import AgeGraphStore
from kg.legal.retrieval.intent_parser import IntentParser
from kg.legal.retrieval.nl2cypher import NL2CypherConverter

logger = logging.getLogger(__name__)
console = Console()

BANNER = """[bold cyan]Legal KG Query CLI[/bold cyan]
[dim]Querying Apache AGE graph (legal_graph).
Type a legal question or 'quit' / Ctrl-C to exit.[/dim]
"""

_HELP_EXAMPLES = [
    "Who are the parties to the Strategic Alliance Agreement?",
    "Which parties indemnify each other?",
    "What is the governing law?",
    "What are the termination clauses?",
    "What is the limitation of liability?",
    "Which contracts lack an indemnity clause?",
    "What are the compliance risks?",
    "List all contracts.",
]


async def _run_query(
    question: str,
    store: AgeGraphStore,
    converter: NL2CypherConverter,
    parser: IntentParser,
    show_cypher: bool,
) -> None:
    question = question.strip()
    if not question:
        return

    match = parser.parse(question)
    cypher = await converter.convert(question)

    if show_cypher:
        console.print(f"\n[dim]Intent:[/dim] [yellow]{match.intent}[/yellow]  "
                      f"[dim]params=[/dim]{match.params}")
        console.print(f"[dim]Cypher:[/dim] [italic]{cypher}[/italic]\n")

    try:
        result = await store.run_cypher_query(cypher)
    except Exception as exc:
        console.print(f"[red]Error executing query:[/red] {exc}")
        return

    lines = [ln for ln in result.splitlines() if ln.strip()]
    if not lines:
        console.print("[dim]No results.[/dim]\n")
        return

    # Render as a Rich table if the result is pipe-separated
    if "|" in lines[0]:
        headers = [h.strip() for h in lines[0].split("|")]
        table = Table(*headers, show_header=True, header_style="bold magenta")
        for row_line in lines[1:]:
            cells = [c.strip() for c in row_line.split("|")]
            # pad/trim to header width
            while len(cells) < len(headers):
                cells.append("")
            table.add_row(*cells[: len(headers)])
        console.print(table)
    else:
        for line in lines:
            console.print(line)

    console.print()


async def _interactive(show_cypher: bool) -> None:
    store = AgeGraphStore()
    await store.initialize()
    converter = NL2CypherConverter()
    parser = IntentParser()

    console.print(BANNER)

    table = Table(title="Example questions", show_header=False, padding=(0, 2))
    table.add_column(style="dim")
    for ex in _HELP_EXAMPLES:
        table.add_row(ex)
    console.print(table)
    console.print()

    try:
        while True:
            try:
                question = input("❯ ").strip()
            except EOFError:
                break
            if question.lower() in {"quit", "exit", "q"}:
                break
            await _run_query(question, store, converter, parser, show_cypher)
    finally:
        await store.close()


async def _single(question: str, show_cypher: bool) -> None:
    store = AgeGraphStore()
    await store.initialize()
    converter = NL2CypherConverter()
    parser = IntentParser()
    try:
        await _run_query(question, store, converter, parser, show_cypher)
    finally:
        await store.close()


async def _stdin_mode(show_cypher: bool) -> None:
    store = AgeGraphStore()
    await store.initialize()
    converter = NL2CypherConverter()
    parser = IntentParser()
    try:
        for line in sys.stdin:
            q = line.strip()
            if q:
                console.print(Panel(q, style="bold"))
                await _run_query(q, store, converter, parser, show_cypher)
    finally:
        await store.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Query the legal knowledge graph in natural language."
    )
    ap.add_argument("--question", "-q", help="Single question (non-interactive)")
    ap.add_argument("--show-cypher", action="store_true",
                    help="Print intent + Cypher before results")
    ap.add_argument("--stdin", action="store_true",
                    help="Read questions from stdin (one per line)")
    ap.add_argument("--debug", action="store_true",
                    help="Enable DEBUG logging")
    args = ap.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.question:
        asyncio.run(_single(args.question, args.show_cypher))
    elif args.stdin:
        asyncio.run(_stdin_mode(args.show_cypher))
    else:
        asyncio.run(_interactive(args.show_cypher))


if __name__ == "__main__":
    main()
