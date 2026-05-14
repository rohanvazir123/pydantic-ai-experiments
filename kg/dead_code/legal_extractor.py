"""
LLM-based legal entity and relationship extractor.

Module: kg.legal_extractor
============================================

Two-phase design — run the expensive LLM pass once, reuse the result:

  Phase 1 — extract  (LLM calls, run once with a capable model)
    LegalEntityExtractor.extract_all_to_file(documents, output_path)
      Iterates every document, calls the LLM in text windows, and serialises
      the extracted entities + relationships to a JSON file at output_path
      (default: rag/legal/llm_kg_extracted.json).

  Phase 2 — build  (no LLM, fast, repeatable)
    LegalEntityExtractor.build_from_file(file_path)
      Reads the saved JSON and upserts everything into the KG backend
      (PgGraphStore or AgeGraphStore) — same pattern as build_cuad_kg() but
      driven by LLM output rather than CUAD annotations.

CLI
---
    # Phase 1 — extract once (requires LLM; use GPT-4o for best results)
    python -m kg.legal_extractor --extract
    python -m kg.legal_extractor --extract --output rag/legal/llm_kg_extracted.json --limit 20

    # Phase 2 — build KG from saved file (no LLM)
    python -m kg.legal_extractor --build
    python -m kg.legal_extractor --build --input rag/legal/llm_kg_extracted.json

Inline usage
------------
    from kg import create_kg_store, LegalEntityExtractor

    store = create_kg_store()
    await store.initialize()
    extractor = LegalEntityExtractor(store)

    # One-shot (extract + store in a single call — used by the ingestion pipeline)
    stats = await extractor.extract_and_store(doc_id, title, markdown_text)

    # Two-phase (recommended for bulk re-runs)
    await extractor.extract_all_to_file(documents, Path("rag/legal/llm_kg_extracted.json"))
    await extractor.build_from_file(Path("rag/legal/llm_kg_extracted.json"))

    await store.close()
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from rag.config.settings import load_settings

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_EXTRACTED_PATH = Path("rag/legal/llm_kg_extracted.json")

# Each window ≈ 3 000 chars; max 5 windows ≈ first 15 000 chars per document.
_MAX_WINDOW_CHARS = 3_000
_MAX_WINDOWS = 5

_SYSTEM_PROMPT = """
You are a legal contract analysis expert. Extract entities and relationships
from the contract text provided.

ENTITY TYPES — use exactly these strings:
  Party             contracting parties, organizations, individuals
  Jurisdiction      governing law, state/country for disputes
  Date              effective date, expiry date, notice period, renewal term
  LicenseClause     license grants, sublicensing, perpetual/irrevocable licenses
  TerminationClause termination for cause/convenience, notice to terminate
  RestrictionClause non-compete, non-solicitation, exclusivity, volume restrictions
  IPClause          IP ownership assignment, joint IP, source code escrow
  LiabilityClause   cap on liability, liquidated damages, uncapped liability
  PaymentTerm       fees, royalties, revenue sharing, minimum commitments
  Obligation        duties a party must perform (delivery, reporting, indemnification)
  Clause            any other significant provision not covered above
  Contract          the agreement itself — only if its name/title appears in the text

RELATIONSHIP TYPES — SCREAMING_SNAKE_CASE only:
  PARTY_TO            Party → Contract
  GOVERNED_BY         Contract → Jurisdiction
  GRANTS_LICENSE_TO   Party → Party  (licensor → licensee)
  OWES_OBLIGATION_TO  Party → Party
  ASSIGNS_IP_TO       Party → Party
  CAN_TERMINATE       Party → Contract or TerminationClause
  HAS_LICENSE         Contract → LicenseClause
  HAS_TERMINATION     Contract → TerminationClause
  HAS_RESTRICTION     Contract → RestrictionClause
  HAS_IP_CLAUSE       Contract → IPClause
  HAS_LIABILITY       Contract → LiabilityClause
  HAS_PAYMENT         Contract → PaymentTerm
  HAS_OBLIGATION      Contract → Obligation
  HAS_CLAUSE          Contract → Clause  (fallback)
  EFFECTIVE_ON        Contract → Date    (set metadata.date_type = "effective")
  EXPIRES_ON          Contract → Date    (set metadata.date_type = "expiry")
  HAS_DATE            Contract → Date    (other date types)

Rules:
- Extract only what is explicitly stated — do not infer.
- Entity names must be exact text from the contract.
- Omit a relationship if source or target name does not match an entity you extracted.
- Return empty lists if nothing is extractable from this passage.

Output format — return ONLY a valid JSON object, no explanation, no markdown fences:
{"entities": [{"name": "...", "entity_type": "...", "metadata": {}}], "relationships": [{"source_name": "...", "target_name": "...", "relationship_type": "...", "properties": {}}]}
"""


class LegalEntity(BaseModel):
    name: str = Field(description="Exact text from the contract")
    entity_type: str = Field(description="One of the ENTITY TYPES listed in the prompt")
    metadata: dict[str, Any] = Field(default_factory=dict)


class LegalRelationship(BaseModel):
    source_name: str = Field(description="Source entity name — must match a name in entities")
    target_name: str = Field(description="Target entity name — must match a name in entities")
    relationship_type: str = Field(description="SCREAMING_SNAKE_CASE relationship type")
    properties: dict[str, Any] = Field(default_factory=dict)


class LegalGraphDocument(BaseModel):
    entities: list[LegalEntity] = Field(default_factory=list)
    relationships: list[LegalRelationship] = Field(default_factory=list)


class LegalEntityExtractor:
    """Extracts legal entities and relationships from document text using an LLM.

    Works with both PgGraphStore and AgeGraphStore backends.

    Recommended workflow for bulk extraction:
        1. extractor.extract_all_to_file(docs, path)   # pay LLM cost once
        2. extractor.build_from_file(path)              # rebuild KG anytime, free
    """

    def __init__(self, graph_store: Any) -> None:
        self.graph_store = graph_store
        settings = load_settings()
        # Use dedicated KG extraction settings when set; fall back to main LLM.
        api_key  = settings.kg_llm_api_key  or settings.llm_api_key
        model_id = settings.kg_llm_model    or settings.llm_model
        base_url = settings.kg_llm_base_url or settings.llm_base_url
        provider = OpenAIProvider(base_url=base_url, api_key=api_key)
        model = OpenAIChatModel(model_id, provider=provider)
        self._agent: Agent[None, str] = Agent(
            model,
            system_prompt=_SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_windows(self, content: str) -> list[str]:
        windows: list[str] = []
        for i in range(0, len(content), _MAX_WINDOW_CHARS):
            windows.append(content[i : i + _MAX_WINDOW_CHARS])
            if len(windows) >= _MAX_WINDOWS:
                break
        return windows

    async def _extract_from_text(self, text: str) -> LegalGraphDocument:
        import re
        for attempt in range(5):
            try:
                result = await self._agent.run(f"Contract text:\n\n{text}")
                raw: str = result.output
                # Strip markdown fences if the model added them
                raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
                raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if not match:
                    logger.warning("LLM returned no JSON object")
                    return LegalGraphDocument()
                data = json.loads(match.group())
                return LegalGraphDocument(
                    entities=[LegalEntity(**e) for e in data.get("entities", [])],
                    relationships=[LegalRelationship(**r) for r in data.get("relationships", [])],
                )
            except Exception as exc:
                exc_str = str(exc)
                if "429" in exc_str or "rate_limit_exceeded" in exc_str:
                    wait = min(2 ** attempt, 60)  # 1, 2, 4, 8, 16 … 60 s
                    logger.info("Rate limited — retrying in %ds (attempt %d)", wait, attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                logger.warning("LLM extraction failed for window: %s", exc)
                return LegalGraphDocument()
        logger.warning("LLM extraction abandoned after 5 rate-limit retries")
        return LegalGraphDocument()

    async def _extract_document(
        self, document_id: str, document_title: str, content: str
    ) -> dict[str, Any]:
        """Run LLM extraction over all windows; return a JSON-serialisable record.

        Does not write to the KG — call _store_document_graph() for that.
        """
        windows = self._make_windows(content)
        seen_names: set[str] = set()
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        for window in windows:
            doc = await self._extract_from_text(window)
            for ent in doc.entities:
                if ent.name not in seen_names:
                    entities.append(ent.model_dump())
                    seen_names.add(ent.name)
            for rel in doc.relationships:
                relationships.append(rel.model_dump())

        return {
            "document_id": document_id,
            "document_title": document_title,
            "entities": entities,
            "relationships": relationships,
        }

    async def _store_document_graph(self, record: dict[str, Any]) -> dict[str, int]:
        """Upsert a pre-extracted record into the KG. No LLM calls."""
        document_id: str = record["document_id"]
        document_title: str = record["document_title"]

        contract_eid = await self.graph_store.upsert_entity(
            name=document_title,
            entity_type="Contract",
            document_id=document_id,
        )
        entity_id_cache: dict[str, str] = {document_title: contract_eid}
        total_entities = 0
        total_rels = 0

        for ent in record["entities"]:
            if ent["entity_type"] == "Contract":
                entity_id_cache.setdefault(ent["name"], contract_eid)
                continue
            try:
                eid = await self.graph_store.upsert_entity(
                    name=ent["name"],
                    entity_type=ent["entity_type"],
                    document_id=document_id,
                    metadata=ent.get("metadata", {}),
                )
                entity_id_cache[ent["name"]] = eid
                total_entities += 1
            except Exception as exc:
                logger.debug("Failed to upsert entity %r: %s", ent["name"], exc)

        for rel in record["relationships"]:
            src_id = entity_id_cache.get(rel["source_name"])
            tgt_id = entity_id_cache.get(rel["target_name"])
            if not src_id or not tgt_id:
                continue
            try:
                await self.graph_store.add_relationship(
                    source_id=src_id,
                    target_id=tgt_id,
                    relationship_type=rel["relationship_type"],
                    document_id=document_id,
                    properties=rel.get("properties", {}),
                )
                total_rels += 1
            except Exception as exc:
                logger.debug("Failed to add relationship %r: %s", rel["relationship_type"], exc)

        return {"entities": total_entities, "relationships": total_rels}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract_and_store(
        self,
        document_id: str,
        document_title: str,
        content: str,
    ) -> dict[str, int]:
        """Extract and immediately persist to the KG (used by the ingestion pipeline).

        Returns {"entities": N, "relationships": N, "windows": N}.
        """
        record = await self._extract_document(document_id, document_title, content)
        stats = await self._store_document_graph(record)
        windows = len(self._make_windows(content))
        logger.info(
            "KG extraction complete: doc=%r windows=%d entities=%d relationships=%d",
            document_title, windows, stats["entities"], stats["relationships"],
        )
        return {**stats, "windows": windows}

    async def extract_all_to_file(
        self,
        documents: list[dict[str, Any]],
        output_path: Path = DEFAULT_EXTRACTED_PATH,
        limit: int | None = None,
    ) -> dict[str, int]:
        """Run LLM extraction on every document and save results to JSON.

        documents — list of {"id": str, "title": str, "content": str}
        Does NOT write to the KG; call build_from_file() for that.

        Returns {"documents": N, "entities": N, "relationships": N}.
        """
        if limit:
            documents = documents[:limit]

        records: list[dict[str, Any]] = []

        with Progress(
            TextColumn("[cyan]Extracting"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("documents", total=len(documents))
            for doc in documents:
                record = await self._extract_document(
                    doc["id"], doc["title"], doc.get("content") or ""
                )
                records.append(record)
                progress.advance(task)
                await asyncio.sleep(2)  # stay inside 30K TPM limit

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        total_entities = sum(len(r["entities"]) for r in records)
        total_rels = sum(len(r["relationships"]) for r in records)
        logger.info(
            "Saved %d document records to %s (entities=%d, relationships=%d)",
            len(records), output_path, total_entities, total_rels,
        )
        return {
            "documents": len(records),
            "entities": total_entities,
            "relationships": total_rels,
        }

    async def build_from_file(
        self,
        file_path: Path = DEFAULT_EXTRACTED_PATH,
    ) -> dict[str, int]:
        """Read pre-extracted JSON and populate the KG. No LLM calls.

        Mirrors build_cuad_kg() but driven by LLM output instead of
        CUAD annotations.

        Returns {"documents": N, "entities": N, "relationships": N}.
        """
        with open(file_path, encoding="utf-8") as f:
            records: list[dict[str, Any]] = json.load(f)

        total_entities = 0
        total_rels = 0

        with Progress(
            TextColumn("[cyan]Building KG"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("records", total=len(records))
            for record in records:
                stats = await self._store_document_graph(record)
                total_entities += stats["entities"]
                total_rels += stats["relationships"]
                progress.advance(task)

        logger.info(
            "KG build from file complete: documents=%d entities=%d relationships=%d",
            len(records), total_entities, total_rels,
        )
        return {
            "documents": len(records),
            "entities": total_entities,
            "relationships": total_rels,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _fetch_documents(limit: int | None) -> list[dict[str, Any]]:
    """Query the documents table and return id/title/content rows."""
    import asyncpg
    settings = load_settings()
    conn = await asyncpg.connect(settings.database_url)
    try:
        if limit:
            rows = await conn.fetch(
                "SELECT id::text, title, content FROM documents"
                " WHERE content IS NOT NULL AND content <> ''"
                " ORDER BY created_at LIMIT $1",
                limit,
            )
        else:
            rows = await conn.fetch(
                "SELECT id::text, title, content FROM documents"
                " WHERE content IS NOT NULL AND content <> ''"
                " ORDER BY created_at"
            )
        return [{"id": r["id"], "title": r["title"], "content": r["content"] or ""} for r in rows]
    finally:
        await conn.close()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-phase LLM extraction for the legal knowledge graph"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--extract",
        action="store_true",
        help="Phase 1: run LLM extraction on all ingested documents and save to JSON",
    )
    mode.add_argument(
        "--build",
        action="store_true",
        help="Phase 2: load saved JSON and populate the KG (no LLM calls)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_EXTRACTED_PATH),
        help=f"Output path for --extract (default: {DEFAULT_EXTRACTED_PATH})",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_EXTRACTED_PATH),
        help=f"Input path for --build (default: {DEFAULT_EXTRACTED_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N documents (for testing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="--extract: overwrite output file if it already exists",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from kg import create_kg_store

    store = create_kg_store()
    await store.initialize()

    try:
        extractor = LegalEntityExtractor(store)

        if args.extract:
            output_path = Path(args.output)
            if output_path.exists() and not args.force:
                console.print(
                    f"[yellow]Output file already exists: {output_path}[/]\n"
                    "[yellow]Use --force to re-run extraction and overwrite it.[/]\n"
                    "[green]To populate the KG from the existing file, run --build instead.[/]"
                )
                return

            console.print("[cyan]Fetching documents from database…[/]")
            documents = await _fetch_documents(args.limit)
            n = len(documents)
            # Estimated cost warning: ~5 windows × 1 500 tokens × n docs
            est_tokens = n * 5 * 1_500
            console.print(
                f"[green]Found {n} documents[/]\n"
                f"[yellow]Estimated token usage: ~{est_tokens:,} tokens "
                f"({n} docs × 5 windows × 1 500 tokens).[/]\n"
                f"[yellow]This uses KG_LLM_MODEL={settings.kg_llm_model or settings.llm_model}. "
                "Press Ctrl-C now to abort, or wait 5 s to continue…[/]"
            )
            await asyncio.sleep(5)

            stats = await extractor.extract_all_to_file(documents, output_path)

            table = Table(title="Extraction Summary", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Documents processed", str(stats["documents"]))
            table.add_row("Entities extracted", str(stats["entities"]))
            table.add_row("Relationships extracted", str(stats["relationships"]))
            table.add_row("Output file", str(output_path))
            console.print(table)

        else:  # --build
            input_path = Path(args.input)
            console.print(f"[cyan]Loading extracted records from[/] {input_path}")
            stats = await extractor.build_from_file(input_path)

            table = Table(title="KG Build Summary", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Documents loaded", str(stats["documents"]))
            table.add_row("Entities upserted", str(stats["entities"]))
            table.add_row("Relationships created", str(stats["relationships"]))
            console.print(table)

    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
