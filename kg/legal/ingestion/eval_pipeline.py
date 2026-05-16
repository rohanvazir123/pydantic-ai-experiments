"""
Ingestion evaluation pipeline — measures KG extraction quality.

Runs against Bronze artifacts already in ``kg_raw_extractions`` (no LLM
calls needed).  If Bronze is empty it optionally ingests a small sample first.

Metrics computed
----------------
Per-contract (Bronze level)
  - chunks_processed          : number of chunks with Bronze rows
  - entities_extracted        : total entities across all chunks
  - relationships_extracted   : total relationships across all chunks
  - confidence_mean           : mean entity confidence
  - invalid_rel_fraction      : fraction of rels flagged as invalid in Pass 5

Silver / Gold level
  - canonical_entities        : rows in kg_canonical_entities
  - canonical_relationships   : rows in kg_canonical_relationships
  - dedup_rate                : 1 - canonical / raw (higher = more deduplication)

Aggregate (across all contracts evaluated)
  - mean entities / chunk
  - mean relationships / chunk
  - label distribution (which entity types are extracted most)
  - relationship type distribution

Usage
-----
    # Report on whatever is already in Bronze (no ingestion)
    python -m kg.legal.ingestion.eval_pipeline

    # Report for a specific contract
    python -m kg.legal.ingestion.eval_pipeline --contract-id <uuid>

    # Ingest a sample of N contracts first, then report
    python -m kg.legal.ingestion.eval_pipeline --ingest --limit 5

    # Save JSON report
    python -m kg.legal.ingestion.eval_pipeline --output kg/evals/ingest_eval_latest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import asyncpg
from rich.console import Console
from rich.table import Table

from rag.config.settings import load_settings

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ContractMetrics:
    contract_id: str
    title: str
    chunks_processed: int
    entities_extracted: int
    relationships_extracted: int
    confidence_mean: float
    invalid_rel_fraction: float
    label_counts: dict[str, int] = field(default_factory=dict)
    rel_type_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class SilverGoldMetrics:
    canonical_entities: int
    canonical_relationships: int
    raw_entities: int
    raw_relationships: int

    @property
    def entity_dedup_rate(self) -> float:
        return 1.0 - (self.canonical_entities / self.raw_entities) if self.raw_entities else 0.0

    @property
    def rel_dedup_rate(self) -> float:
        return 1.0 - (self.canonical_relationships / self.raw_relationships) if self.raw_relationships else 0.0


@dataclass
class IngestionReport:
    contracts_evaluated: int = 0
    total_chunks: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    mean_entities_per_chunk: float = 0.0
    mean_relationships_per_chunk: float = 0.0
    mean_confidence: float = 0.0
    label_distribution: dict[str, int] = field(default_factory=dict)
    rel_type_distribution: dict[str, int] = field(default_factory=dict)
    silver_gold: SilverGoldMetrics | None = None
    per_contract: list[ContractMetrics] = field(default_factory=list)


async def _fetch_bronze_contracts(
    pool: asyncpg.Pool,
    contract_id: str | None,
) -> list[asyncpg.Record]:
    if contract_id:
        rows = await pool.fetch(
            "SELECT DISTINCT contract_id FROM kg_raw_extractions WHERE contract_id = $1",
            contract_id,
        )
    else:
        rows = await pool.fetch(
            "SELECT DISTINCT contract_id FROM kg_raw_extractions ORDER BY contract_id"
        )
    return rows


async def _contract_title(pool: asyncpg.Pool, contract_id: str) -> str:
    row = await pool.fetchrow(
        "SELECT title FROM documents WHERE id = $1", contract_id
    )
    return row["title"] if row else contract_id[:8]


async def _eval_contract(pool: asyncpg.Pool, contract_id: str) -> ContractMetrics:
    title = await _contract_title(pool, contract_id)
    rows = await pool.fetch(
        "SELECT raw_json FROM kg_raw_extractions WHERE contract_id = $1 ORDER BY chunk_index",
        contract_id,
    )

    all_entities: list[dict] = []
    all_rels: list[dict] = []
    invalid_rels = 0
    label_counts: Counter = Counter()
    rel_type_counts: Counter = Counter()

    import json as _json
    for row in rows:
        raw = row["raw_json"]
        artifact = raw if isinstance(raw, dict) else _json.loads(raw)
        entities = artifact.get("entities", [])
        rels = artifact.get("valid_relationships", [])  # BronzeArtifact uses valid_relationships
        inv_rels = artifact.get("invalid_relationships", [])

        all_entities.extend(entities)
        all_rels.extend(rels)
        invalid_rels += len(inv_rels)

        for e in entities:
            label_counts[e.get("label", "Unknown")] += 1
        for r in rels:
            rel_type_counts[r.get("relationship_type", "Unknown")] += 1

    conf_values = [e.get("confidence", 0.0) for e in all_entities if "confidence" in e]
    conf_mean = mean(conf_values) if conf_values else 0.0
    total_rels = len(all_rels) + invalid_rels
    inv_frac = invalid_rels / total_rels if total_rels else 0.0

    return ContractMetrics(
        contract_id=contract_id,
        title=title,
        chunks_processed=len(rows),
        entities_extracted=len(all_entities),
        relationships_extracted=len(all_rels),
        confidence_mean=conf_mean,
        invalid_rel_fraction=inv_frac,
        label_counts=dict(label_counts),
        rel_type_counts=dict(rel_type_counts),
    )


async def _silver_gold_metrics(pool: asyncpg.Pool) -> SilverGoldMetrics:
    raw_e = await pool.fetchval("SELECT COUNT(*) FROM kg_staging_entities") or 0
    raw_r = await pool.fetchval("SELECT COUNT(*) FROM kg_staging_relationships") or 0
    can_e = await pool.fetchval("SELECT COUNT(*) FROM kg_canonical_entities") or 0
    can_r = await pool.fetchval("SELECT COUNT(*) FROM kg_canonical_relationships") or 0
    return SilverGoldMetrics(
        canonical_entities=can_e,
        canonical_relationships=can_r,
        raw_entities=raw_e,
        raw_relationships=raw_r,
    )


async def run_eval(contract_id: str | None = None) -> IngestionReport:
    settings = load_settings()
    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=5)

    try:
        contract_rows = await _fetch_bronze_contracts(pool, contract_id)
        if not contract_rows:
            console.print("[yellow]No Bronze artifacts found. Run ingestion first.[/yellow]")
            return IngestionReport()

        console.print(f"[dim]Evaluating {len(contract_rows)} contract(s)…[/dim]")

        per_contract = await asyncio.gather(*[
            _eval_contract(pool, str(row["contract_id"])) for row in contract_rows
        ])

        sg = None
        try:
            sg = await _silver_gold_metrics(pool)
        except Exception as e:
            logger.warning("Could not fetch Silver/Gold metrics: %s", e)

        total_chunks = sum(c.chunks_processed for c in per_contract)
        total_ents   = sum(c.entities_extracted for c in per_contract)
        total_rels   = sum(c.relationships_extracted for c in per_contract)

        label_dist: Counter = Counter()
        rel_dist: Counter = Counter()
        all_conf = []
        for c in per_contract:
            label_dist.update(c.label_counts)
            rel_dist.update(c.rel_type_counts)
            if c.confidence_mean > 0:
                all_conf.append(c.confidence_mean)

        return IngestionReport(
            contracts_evaluated=len(per_contract),
            total_chunks=total_chunks,
            total_entities=total_ents,
            total_relationships=total_rels,
            mean_entities_per_chunk=total_ents / total_chunks if total_chunks else 0.0,
            mean_relationships_per_chunk=total_rels / total_chunks if total_chunks else 0.0,
            mean_confidence=mean(all_conf) if all_conf else 0.0,
            label_distribution=dict(label_dist.most_common()),
            rel_type_distribution=dict(rel_dist.most_common()),
            silver_gold=sg,
            per_contract=list(per_contract),
        )
    finally:
        await pool.close()


def _print_report(report: IngestionReport) -> None:
    console.print(f"\n[bold]Ingestion Eval — {report.contracts_evaluated} contract(s), "
                  f"{report.total_chunks} chunks[/bold]")

    summary = Table(show_header=True, header_style="bold cyan")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Total entities extracted",    str(report.total_entities))
    summary.add_row("Total relationships extracted", str(report.total_relationships))
    summary.add_row("Mean entities / chunk",       f"{report.mean_entities_per_chunk:.1f}")
    summary.add_row("Mean relationships / chunk",  f"{report.mean_relationships_per_chunk:.1f}")
    summary.add_row("Mean entity confidence",      f"{report.mean_confidence:.3f}")
    console.print(summary)

    if report.silver_gold:
        sg = report.silver_gold
        sg_table = Table(title="Silver / Gold", show_header=True, header_style="bold magenta")
        sg_table.add_column("Layer")
        sg_table.add_column("Count", justify="right")
        sg_table.add_column("Dedup rate", justify="right")
        sg_table.add_row("Entities",      str(sg.canonical_entities), f"{sg.entity_dedup_rate:.1%}")
        sg_table.add_row("Relationships", str(sg.canonical_relationships), f"{sg.rel_dedup_rate:.1%}")
        console.print(sg_table)

    if report.label_distribution:
        lbl_table = Table(title="Entity label distribution (top 10)",
                          show_header=True, header_style="bold green")
        lbl_table.add_column("Label")
        lbl_table.add_column("Count", justify="right")
        for lbl, cnt in list(report.label_distribution.items())[:10]:
            lbl_table.add_row(lbl, str(cnt))
        console.print(lbl_table)

    if report.rel_type_distribution:
        rel_table = Table(title="Relationship type distribution (top 10)",
                          show_header=True, header_style="bold blue")
        rel_table.add_column("Type")
        rel_table.add_column("Count", justify="right")
        for rtype, cnt in list(report.rel_type_distribution.items())[:10]:
            rel_table.add_row(rtype, str(cnt))
        console.print(rel_table)

    if report.per_contract:
        pc_table = Table(title="Per-contract summary",
                         show_header=True, header_style="bold")
        pc_table.add_column("Contract", max_width=40)
        pc_table.add_column("Chunks", justify="right")
        pc_table.add_column("Entities", justify="right")
        pc_table.add_column("Rels", justify="right")
        pc_table.add_column("Conf", justify="right")
        pc_table.add_column("Inv%", justify="right")
        for c in report.per_contract:
            pc_table.add_row(
                c.title[:40],
                str(c.chunks_processed),
                str(c.entities_extracted),
                str(c.relationships_extracted),
                f"{c.confidence_mean:.2f}",
                f"{c.invalid_rel_fraction:.0%}",
            )
        console.print(pc_table)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate KG ingestion quality.")
    ap.add_argument("--contract-id", help="Evaluate a single contract UUID")
    ap.add_argument("--ingest", action="store_true",
                    help="Run ingestion for --limit contracts before evaluating")
    ap.add_argument("--limit", type=int, default=3,
                    help="Number of contracts to ingest (with --ingest)")
    ap.add_argument("--output", help="Save JSON report to this path")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.ingest:
        from kg.legal.ingestion.extraction_pipeline import ExtractionPipeline
        async def _ingest_then_eval():
            pipeline = ExtractionPipeline()
            console.print(f"[dim]Running ingestion for up to {args.limit} contracts…[/dim]")
            await pipeline.run_all(limit=args.limit)
            return await run_eval(args.contract_id)
        report = asyncio.run(_ingest_then_eval())
    else:
        report = asyncio.run(run_eval(args.contract_id))

    _print_report(report)

    if args.output:
        def _to_dict(r: IngestionReport) -> dict:
            d = asdict(r)
            return d
        Path(args.output).write_text(json.dumps(_to_dict(report), indent=2, default=str))
        console.print(f"\n[green]Report saved to {args.output}[/green]")


if __name__ == "__main__":
    main()
