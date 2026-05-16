"""
Build an Apache AGE knowledge graph from CUAD evaluation annotations.

Module: kg.legal.cuad_kg_ingest
==========================================

Reads ``rag/legal/cuad_eval.json`` and writes entities + relationships
directly to AgeGraphStore. Document ID lookups use a plain asyncpg pool
against the main PostgreSQL DB (the one that holds the ``documents`` table).

All clause-type → entity-type and entity-type → relationship-type mappings
come from cuad_ontology.py — the single source of truth for KG ontology.

CLI
---
    python -m kg.legal.cuad_kg_ingest
    python -m kg.legal.cuad_kg_ingest --eval-path rag/legal/cuad_eval.json --limit 100
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import asyncpg
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from kg.age_graph_store import AgeGraphStore
from kg.legal.cuad_ontology import entity_type_for, relationship_type_for

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_EVAL_PATH = Path("rag/legal/cuad_eval.json")


async def _get_document_id(
    pool: asyncpg.Pool,
    contract_title: str,
    cache: dict[str, str | None],
) -> str | None:
    """Look up a document UUID by title, caching results across calls."""
    if contract_title in cache:
        return cache[contract_title]
    normalized = contract_title.replace("\\_", "_")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM documents "
            "WHERE REPLACE(title, '\\_', '_') = $1 LIMIT 1",
            normalized,
        )
    doc_id = str(row["id"]) if row else None
    cache[contract_title] = doc_id
    return doc_id


async def build_cuad_kg(
    store: AgeGraphStore,
    doc_pool: asyncpg.Pool,
    eval_path: Path = DEFAULT_EVAL_PATH,
    limit: int | None = None,
) -> dict[str, int]:
    """Ingest CUAD eval annotations into the AGE graph.

    Args:
        store:     Initialized AgeGraphStore for graph writes.
        doc_pool:  asyncpg pool connected to the main PostgreSQL DB (documents table).
        eval_path: Path to cuad_eval.json.
        limit:     Cap on eval pairs to process (for testing).

    Returns:
        {"entities": N, "relationships": N, "skipped": N}
    """
    with open(eval_path, encoding="utf-8") as f:
        pairs: list[dict[str, Any]] = json.load(f)

    if limit:
        pairs = pairs[:limit]

    doc_id_cache: dict[str, str | None] = {}
    entities_created = 0
    relationships_created = 0
    skipped = 0

    with Progress(
        TextColumn("[cyan]Building KG"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("pairs", total=len(pairs))

        for pair in pairs:
            progress.advance(task)

            contract_title = pair["contract_title"]
            question_type = pair["question_type"]
            answers = pair.get("answers", [])

            if not answers:
                skipped += 1
                continue

            doc_id = await _get_document_id(doc_pool, contract_title, doc_id_cache)
            if doc_id is None:
                logger.debug("Document not found in DB: %r", contract_title)
                skipped += 1
                continue

            entity_type = entity_type_for(question_type)
            rel_type = relationship_type_for(entity_type)

            contract_eid = await store.upsert_entity(
                name=contract_title,
                entity_type="Contract",
                document_id=doc_id,
                metadata={"contract_type": pair.get("contract_type", "")},
            )
            entities_created += 1

            for answer_text in answers:
                if not answer_text or not answer_text.strip():
                    continue

                eid = await store.upsert_entity(
                    name=answer_text.strip(),
                    entity_type=entity_type,
                    document_id=doc_id,
                    metadata={
                        "question_type": question_type,
                        "contract_type": pair.get("contract_type", ""),
                    },
                )
                entities_created += 1

                await store.add_relationship(
                    source_id=eid,
                    target_id=contract_eid,
                    relationship_type=rel_type,
                    document_id=doc_id,
                    properties={"question_type": question_type},
                )
                relationships_created += 1

    return {
        "entities": entities_created,
        "relationships": relationships_created,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Apache AGE knowledge graph from CUAD eval annotations"
    )
    parser.add_argument(
        "--eval-path",
        default=str(DEFAULT_EVAL_PATH),
        help="Path to cuad_eval.json (default: rag/legal/cuad_eval.json)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from rag.config.settings import load_settings
    from kg.age_graph_store import _age_init

    settings = load_settings()

    store = AgeGraphStore()
    await store.initialize()

    doc_pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=3)

    console.print(f"[cyan]Loading eval pairs from[/] {args.eval_path}")

    try:
        stats = await build_cuad_kg(
            store=store,
            doc_pool=doc_pool,
            eval_path=Path(args.eval_path),
            limit=args.limit,
        )
    finally:
        await store.close()
        await doc_pool.close()

    table = Table(title="CUAD Knowledge Graph Build Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Entities created / updated", str(stats["entities"]))
    table.add_row("Relationships created", str(stats["relationships"]))
    table.add_row("Skipped (no doc or no answer)", str(stats["skipped"]))
    console.print(table)


if __name__ == "__main__":
    asyncio.run(main())
