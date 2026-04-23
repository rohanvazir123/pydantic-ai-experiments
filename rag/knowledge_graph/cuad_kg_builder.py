"""
Build a PostgreSQL knowledge graph from CUAD evaluation annotations.

Module: rag.knowledge_graph.cuad_kg_builder
============================================

Reads ``rag/legal/cuad_eval.json`` (produced by the CUAD ingestion script)
and populates ``kg_entities`` / ``kg_relationships`` without any LLM calls.
Each Q&A pair already tells us the clause type and the extracted answer text,
so entity extraction is deterministic and instant.

Entity-type mapping (CUAD question_type → kg entity_type)
----------------------------------------------------------
    Parties                    → Party
    Governing Law              → Jurisdiction
    Agreement Date             → Date
    Effective Date             → Date
    Expiration Date            → Date
    License Grant              → LicenseClause
    Non-Transferable License   → LicenseClause
    Irrevocable or Perpetual License → LicenseClause
    Unlimited/All-You-Can-Eat License → LicenseClause
    Affiliate License-Licensor → LicenseClause
    Affiliate License-Licensee → LicenseClause
    Termination for Convenience → TerminationClause
    Termination for Cause      → TerminationClause
    Non-Compete                → RestrictionClause
    No-Solicit of Customers    → RestrictionClause
    No-Solicit of Employees    → RestrictionClause
    Exclusivity                → RestrictionClause
    IP Ownership Assignment    → IPClause
    Joint IP Ownership         → IPClause
    Source Code Escrow         → IPClause
    Change of Control          → Clause
    Anti-Assignment            → Clause
    Cap on Liability           → LiabilityClause
    Uncapped Liability         → LiabilityClause
    Liquidated Damages         → LiabilityClause
    (all others)               → Clause

Relationship types
------------------
    Party            → PARTY_TO          → Contract
    Jurisdiction     → GOVERNED_BY_LAW   → Contract
    Date             → DATE_OF           → Contract   (with properties.date_type)
    LicenseClause    → HAS_LICENSE       → Contract
    TerminationClause → HAS_TERMINATION  → Contract
    RestrictionClause → HAS_RESTRICTION  → Contract
    IPClause         → HAS_IP_CLAUSE     → Contract
    LiabilityClause  → HAS_LIABILITY     → Contract
    Clause           → HAS_CLAUSE        → Contract

Usage
-----
    from rag.knowledge_graph.cuad_kg_builder import CuadKgBuilder
    from rag.knowledge_graph.pg_graph_store import PgGraphStore

    store = PgGraphStore()
    await store.initialize()

    builder = CuadKgBuilder(store)
    stats = await builder.build(eval_path=Path("rag/legal/cuad_eval.json"))
    print(stats)  # {"entities": 1234, "relationships": 987, "skipped": 45}

    await store.close()

CLI
---
    python -m rag.knowledge_graph.cuad_kg_builder
    python -m rag.knowledge_graph.cuad_kg_builder --eval-path rag/legal/cuad_eval.json --limit 100
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from rag.knowledge_graph.pg_graph_store import PgGraphStore

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_EVAL_PATH = Path("rag/legal/cuad_eval.json")

# ---------------------------------------------------------------------------
# Clause-type → entity-type mapping
# ---------------------------------------------------------------------------

ENTITY_TYPE_MAP: dict[str, str] = {
    "Parties": "Party",
    "Governing Law": "Jurisdiction",
    "Agreement Date": "Date",
    "Effective Date": "Date",
    "Expiration Date": "Date",
    "Renewal Term": "Date",
    "License Grant": "LicenseClause",
    "Non-Transferable License": "LicenseClause",
    "Irrevocable or Perpetual License": "LicenseClause",
    "Unlimited/All-You-Can-Eat License": "LicenseClause",
    "Affiliate License-Licensor": "LicenseClause",
    "Affiliate License-Licensee": "LicenseClause",
    "Termination for Convenience": "TerminationClause",
    "Termination for Cause": "TerminationClause",
    "Non-Compete": "RestrictionClause",
    "No-Solicit of Customers": "RestrictionClause",
    "No-Solicit of Employees": "RestrictionClause",
    "Exclusivity": "RestrictionClause",
    "Competitive Restriction Exception": "RestrictionClause",
    "IP Ownership Assignment": "IPClause",
    "Joint IP Ownership": "IPClause",
    "Source Code Escrow": "IPClause",
    "Cap on Liability": "LiabilityClause",
    "Uncapped Liability": "LiabilityClause",
    "Liquidated Damages": "LiabilityClause",
    "Change of Control": "Clause",
    "Anti-Assignment": "Clause",
    "Revenue/Profit Sharing": "Clause",
    "Price Restrictions": "Clause",
    "Minimum Commitment": "Clause",
    "Volume Restriction": "Clause",
    "Post-Termination Services": "Clause",
    "Audit Rights": "Clause",
    "Warranty Duration": "Clause",
    "Insurance": "Clause",
    "Covenant Not to Sue": "Clause",
    "Third Party Beneficiary": "Clause",
    "Most Favored Nation": "Clause",
    "Non-Disparagement": "Clause",
    "Notice Period to Terminate Renewal": "Clause",
}

RELATIONSHIP_MAP: dict[str, str] = {
    "Party": "PARTY_TO",
    "Jurisdiction": "GOVERNED_BY_LAW",
    "Date": "HAS_DATE",
    "LicenseClause": "HAS_LICENSE",
    "TerminationClause": "HAS_TERMINATION",
    "RestrictionClause": "HAS_RESTRICTION",
    "IPClause": "HAS_IP_CLAUSE",
    "LiabilityClause": "HAS_LIABILITY",
    "Clause": "HAS_CLAUSE",
    "Contract": "IS_CONTRACT",
}


def entity_type_for(question_type: str) -> str:
    return ENTITY_TYPE_MAP.get(question_type, "Clause")


def relationship_type_for(entity_type: str) -> str:
    return RELATIONSHIP_MAP.get(entity_type, "HAS_CLAUSE")


class CuadKgBuilder:
    """Populates kg_entities / kg_relationships from cuad_eval.json annotations."""

    def __init__(self, store: PgGraphStore) -> None:
        self.store = store
        self._doc_id_cache: dict[str, str | None] = {}

    async def _get_document_id(self, contract_title: str) -> str | None:
        """Look up the document UUID by title (cached per run)."""
        if contract_title in self._doc_id_cache:
            return self._doc_id_cache[contract_title]

        assert self.store.pool
        async with self.store.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM documents WHERE title = $1 LIMIT 1",
                contract_title,
            )
        doc_id = str(row["id"]) if row else None
        self._doc_id_cache[contract_title] = doc_id
        return doc_id

    async def build(
        self,
        eval_path: Path = DEFAULT_EVAL_PATH,
        limit: int | None = None,
    ) -> dict[str, int]:
        """
        Read cuad_eval.json and populate the KG tables.

        Args:
            eval_path: Path to cuad_eval.json.
            limit: Optional max number of eval pairs to process (for testing).

        Returns:
            {"entities": N, "relationships": N, "skipped": N}
        """
        with open(eval_path, encoding="utf-8") as f:
            pairs: list[dict[str, Any]] = json.load(f)

        if limit:
            pairs = pairs[:limit]

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

                doc_id = await self._get_document_id(contract_title)
                if doc_id is None:
                    logger.debug(f"Document not found in DB: {contract_title!r}")
                    skipped += 1
                    continue

                entity_type = entity_type_for(question_type)
                rel_type = relationship_type_for(entity_type)

                # Upsert a Contract node for the document itself (once per doc)
                contract_eid = await self.store.upsert_entity(
                    name=contract_title,
                    entity_type="Contract",
                    document_id=doc_id,
                    metadata={"contract_type": pair.get("contract_type", "")},
                )
                entities_created += 1

                # One entity per answer text
                for answer_text in answers:
                    if not answer_text or not answer_text.strip():
                        continue

                    eid = await self.store.upsert_entity(
                        name=answer_text.strip(),
                        entity_type=entity_type,
                        document_id=doc_id,
                        metadata={
                            "question_type": question_type,
                            "contract_type": pair.get("contract_type", ""),
                        },
                    )
                    entities_created += 1

                    await self.store.add_relationship(
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
        description="Build PostgreSQL knowledge graph from CUAD eval annotations"
    )
    parser.add_argument(
        "--eval-path",
        default=str(DEFAULT_EVAL_PATH),
        help="Path to cuad_eval.json (default: rag/legal/cuad_eval.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N eval pairs (for testing)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    console.print(f"[cyan]Loading eval pairs from[/] {args.eval_path}")
    store = PgGraphStore()
    await store.initialize()

    try:
        builder = CuadKgBuilder(store)
        stats = await builder.build(
            eval_path=Path(args.eval_path),
            limit=args.limit,
        )
    finally:
        await store.close()

    table = Table(title="CUAD Knowledge Graph Build Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Entities created / updated", str(stats["entities"]))
    table.add_row("Relationships created", str(stats["relationships"]))
    table.add_row("Skipped (no doc or no answer)", str(stats["skipped"]))
    console.print(table)

    kg_stats = await _print_kg_stats()
    console.print(kg_stats)


async def _print_kg_stats() -> str:
    store = PgGraphStore()
    await store.initialize()
    try:
        s = await store.get_graph_stats()
    finally:
        await store.close()

    lines = [
        f"\n[green]KG Stats:[/]",
        f"  Total entities:      {s['total_entities']}",
        f"  Total relationships: {s['total_relationships']}",
        "  By entity type:",
    ]
    for k, v in s["entities_by_type"].items():
        lines.append(f"    {k}: {v}")
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())
