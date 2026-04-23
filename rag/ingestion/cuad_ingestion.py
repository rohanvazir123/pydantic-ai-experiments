"""
CUAD (Contract Understanding Atticus Dataset) ingestion.

Module: rag.ingestion.cuad_ingestion
=====================================

Loads CUAD_v1.json, writes 510 commercial contracts as Markdown files into
``rag/documents/legal/``, saves the 41-question evaluation Q&A pairs to
``rag/legal/cuad_eval.json``, then runs the existing ingestion pipeline so
all contracts are chunked, embedded, and stored in PostgreSQL.

Dataset
-------
- 510 commercial contracts (NDAs, MSAs, distribution, co-branding, …)
- 41 expert-annotated question types per contract (parties, dates, obligations,
  termination, liability, governing law, …)
- 20,910 total Q&A pairs; 6,702 answered; 14,208 not applicable

Download
--------
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="theatticusproject/cuad",
        filename="CUAD_v1/CUAD_v1.json",
        repo_type="dataset",
        local_dir="C:/hf/cuad",
    )

Usage
-----
    # Dry run — extract files + eval pairs, no DB ingestion
    python -m rag.ingestion.cuad_ingestion --dry-run

    # Test run — ingest first 10 contracts
    python -m rag.ingestion.cuad_ingestion --limit 10

    # Full run — all 510 contracts (takes a while)
    python -m rag.ingestion.cuad_ingestion

    # Incremental — skip already-ingested contracts
    python -m rag.ingestion.cuad_ingestion --no-clean
"""

import argparse
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from rag.ingestion.pipeline import create_pipeline

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_CUAD_JSON = Path("C:/hf/cuad/CUAD_v1/CUAD_v1.json")
DEFAULT_OUTPUT_DIR = Path("rag/documents/legal")
DEFAULT_EVAL_PATH = Path("rag/legal/cuad_eval.json")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CuadQA:
    """A single Q&A pair from CUAD."""

    question: str
    question_type: str          # extracted from the question text, e.g. "Parties"
    answers: list[str]
    answer_starts: list[int]
    is_impossible: bool


@dataclass
class CuadContract:
    """A single CUAD contract with metadata and Q&A pairs."""

    title: str
    contract_type: str
    source_id: str              # original CUAD title string (unique)
    text: str
    qas: list[CuadQA] = field(default_factory=list)

    @property
    def safe_filename(self) -> str:
        """Filesystem-safe filename derived from source_id."""
        name = re.sub(r"[^\w\-]", "_", self.source_id)
        return f"{name[:120]}.md"       # cap at 120 chars to avoid Windows MAX_PATH


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_QUESTION_TYPE_RE = re.compile(r'"([^"]+)"')


def _extract_question_type(question: str) -> str:
    """Extract the clause type from a CUAD question string.

    CUAD questions follow the pattern:
        'Highlight the parts ... related to "Clause Type" that should ...'
    """
    m = _QUESTION_TYPE_RE.search(question)
    return m.group(1) if m else "Unknown"


def _extract_contract_type(title: str) -> str:
    """Extract contract type from the CUAD title.

    Titles follow: COMPANYNAME_DATE_FILING_EXHIBIT_ID_CONTRACT TYPE
    The contract type is the last '_'-separated segment.
    """
    parts = title.rsplit("_", 1)
    if len(parts) == 2 and len(parts[-1]) > 3:
        return parts[-1].title()
    # Fallback: look for all-caps words at the end
    m = re.search(r"([A-Z][A-Z\s]+[A-Z])$", title)
    return m.group(1).title() if m else "Commercial Contract"


def load_cuad(json_path: Path = DEFAULT_CUAD_JSON) -> list[CuadContract]:
    """Load and parse CUAD_v1.json into a list of CuadContract objects."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    contracts = []
    for item in data["data"]:
        title = item["title"]
        para = item["paragraphs"][0]          # CUAD has exactly 1 paragraph per contract
        text = para["context"]

        qas = []
        for qa in para["qas"]:
            qas.append(
                CuadQA(
                    question=qa["question"],
                    question_type=_extract_question_type(qa["question"]),
                    answers=[a["text"] for a in qa["answers"]],
                    answer_starts=[a["answer_start"] for a in qa["answers"]],
                    is_impossible=qa["is_impossible"],
                )
            )

        contracts.append(
            CuadContract(
                title=title,
                contract_type=_extract_contract_type(title),
                source_id=title,
                text=text,
                qas=qas,
            )
        )

    return contracts


# ---------------------------------------------------------------------------
# File extraction
# ---------------------------------------------------------------------------


def write_contract_files(
    contracts: list[CuadContract],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """Write each contract as a Markdown file with a YAML-style header comment.

    Returns list of written file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with Progress(
        TextColumn("[cyan]Writing files"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("files", total=len(contracts))

        for contract in contracts:
            path = output_dir / contract.safe_filename
            content = (
                f"# {contract.title}\n\n"
                f"<!-- contract_type: {contract.contract_type} -->\n\n"
                f"{contract.text}"
            )
            path.write_text(content, encoding="utf-8")
            written.append(path)
            progress.advance(task)

    return written


# ---------------------------------------------------------------------------
# Eval dataset
# ---------------------------------------------------------------------------


def save_eval_pairs(
    contracts: list[CuadContract],
    output_path: Path = DEFAULT_EVAL_PATH,
) -> int:
    """Save all Q&A pairs (answered only) to a JSON file for retrieval evaluation.

    Returns the number of Q&A pairs saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = []
    for contract in contracts:
        for qa in contract.qas:
            if not qa.is_impossible and qa.answers:
                pairs.append(
                    {
                        "contract_title": contract.title,
                        "contract_type": contract.contract_type,
                        "question_type": qa.question_type,
                        "question": qa.question,
                        "answers": qa.answers,
                        "answer_starts": qa.answer_starts,
                    }
                )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    return len(pairs)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


async def run_ingestion(
    documents_folder: str,
    clean: bool = False,
    chunk_size: int = 1000,
    max_tokens: int = 512,
) -> dict:
    """Run the existing ingestion pipeline on the legal documents folder."""
    pipeline = create_pipeline(
        documents_folder=documents_folder,
        clean=clean,
        chunk_size=chunk_size,
        max_tokens=max_tokens,
    )
    try:
        await pipeline.initialize()
        results = await pipeline.ingest_documents()
    finally:
        await pipeline.close()

    return {
        "documents": len(results),
        "chunks": sum(r.chunks_created for r in results),
        "errors": sum(len(r.errors) for r in results),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_summary(contracts: int, files: int, eval_pairs: int, ingest: dict | None) -> None:
    table = Table(title="CUAD Ingestion Summary", show_header=True)
    table.add_column("Step", style="cyan")
    table.add_column("Result", style="green")

    table.add_row("Contracts parsed", str(contracts))
    table.add_row("Contract files written", str(files))
    table.add_row("Eval Q&A pairs saved", str(eval_pairs))

    if ingest:
        table.add_row("Documents ingested", str(ingest["documents"]))
        table.add_row("Chunks created", str(ingest["chunks"]))
        table.add_row("Ingest errors", str(ingest["errors"]))

    console.print(table)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CUAD legal contracts into PostgreSQL")
    parser.add_argument(
        "--cuad-json",
        default=str(DEFAULT_CUAD_JSON),
        help="Path to CUAD_v1.json (default: C:/hf/cuad/CUAD_v1/CUAD_v1.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to write contract .md files (default: rag/documents/legal)",
    )
    parser.add_argument(
        "--eval-path",
        default=str(DEFAULT_EVAL_PATH),
        help="Where to save eval Q&A pairs (default: rag/legal/cuad_eval.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N contracts (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract files and eval pairs but skip DB ingestion",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Incremental ingestion — skip already-ingested contracts",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per chunk (default: 512)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # 1. Load
    console.print(f"[cyan]Loading CUAD from[/] {args.cuad_json}")
    contracts = load_cuad(Path(args.cuad_json))
    if args.limit:
        contracts = contracts[: args.limit]
        console.print(f"[yellow]Limiting to {args.limit} contracts[/]")
    console.print(f"[green]Loaded {len(contracts)} contracts[/]")

    # 2. Write contract files
    console.print(f"\n[cyan]Writing contract files to[/] {args.output_dir}")
    written = write_contract_files(contracts, Path(args.output_dir))

    # 3. Save eval Q&A pairs
    console.print(f"\n[cyan]Saving eval Q&A pairs to[/] {args.eval_path}")
    eval_count = save_eval_pairs(contracts, Path(args.eval_path))
    console.print(f"[green]Saved {eval_count} answered Q&A pairs[/]")

    # 4. Ingest into PostgreSQL
    ingest_result = None
    if not args.dry_run:
        console.print(f"\n[cyan]Running ingestion pipeline on[/] {args.output_dir}")
        t0 = time.time()
        ingest_result = await run_ingestion(
            documents_folder=args.output_dir,
            clean=not args.no_clean,
            chunk_size=args.chunk_size,
            max_tokens=args.max_tokens,
        )
        elapsed = time.time() - t0
        console.print(f"[green]Ingestion complete in {elapsed:.1f}s[/]")
    else:
        console.print("\n[yellow]Dry run — skipping DB ingestion[/]")

    # 5. Summary
    console.print()
    _print_summary(len(contracts), len(written), eval_count, ingest_result)


if __name__ == "__main__":
    asyncio.run(main())
