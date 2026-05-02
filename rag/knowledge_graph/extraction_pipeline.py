"""
Multi-pass legal knowledge graph extraction pipeline.

Module: rag.knowledge_graph.extraction_pipeline
===============================================

Bronze / Silver / Gold medallion architecture.  See docs/KG_PIPELINE.md.

CLI
---
    # Full pipeline (Bronze + Silver + Gold) for one contract
    python -m rag.knowledge_graph.extraction_pipeline --contract-id <uuid>

    # Full pipeline for all contracts
    python -m rag.knowledge_graph.extraction_pipeline --all [--limit N]

    # Replay Silver + Gold from existing Bronze (no LLM calls)
    python -m rag.knowledge_graph.extraction_pipeline --project --contract-id <uuid>
    python -m rag.knowledge_graph.extraction_pipeline --project --all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from rag.config.settings import load_settings
from rag.knowledge_graph.pg_graph_store import _normalize

logger = logging.getLogger(__name__)
console = Console()

ONTOLOGY_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Pydantic models for extraction outputs
# ---------------------------------------------------------------------------

VALID_LABELS = frozenset({
    "Contract", "Section", "Clause", "Party", "Jurisdiction",
    "EffectiveDate", "ExpirationDate", "RenewalTerm", "LiabilityClause",
    "IndemnityClause", "PaymentTerm", "ConfidentialityClause",
    "TerminationClause", "GoverningLawClause", "Obligation",
    "Risk", "Amendment", "ReferenceDocument",
})

VALID_REL_TYPES = frozenset({
    "SIGNED_BY", "GOVERNED_BY", "INDEMNIFIES", "HAS_TERMINATION", "HAS_RENEWAL",
    "HAS_PAYMENT_TERM", "REFERENCES", "AMENDS", "SUPERCEDES", "REPLACES",
    "OBLIGATES", "LIMITS_LIABILITY", "DISCLOSES_TO", "HAS_CLAUSE",
    "HAS_SECTION", "HAS_CHUNK", "ATTACHES", "INCORPORATES_BY_REFERENCE",
    "INCREASES_RISK_FOR", "CAUSES",
})


class ExtractedEntity(BaseModel):
    entity_id: str
    label: str
    canonical_name: str
    text_span: str
    confidence: float = Field(ge=0.0, le=1.0)

    def safe_label(self) -> str:
        return self.label if self.label in VALID_LABELS else "Clause"


class ExtractedRelationship(BaseModel):
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    evidence_text: str
    confidence: float = Field(ge=0.0, le=1.0)

    def safe_rel_type(self) -> str | None:
        return self.relationship_type if self.relationship_type in VALID_REL_TYPES else None


class HierarchyNode(BaseModel):
    node_id: str
    node_type: str
    title: str
    sequence_number: int


class HierarchyEdge(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str


class CrossContractRef(BaseModel):
    source_contract_id: str
    target_document_name: str
    relationship_type: str
    evidence_text: str
    confidence: float = Field(ge=0.0, le=1.0)


class InvalidRelationship(BaseModel):
    relationship_id: str
    reason: str


class BronzeArtifact(BaseModel):
    artifact_id: str = Field(default_factory=lambda: str(_uuid.uuid4()))
    contract_id: str
    chunk_index: int
    chunk_text: str
    model_version: str
    ontology_version: str = ONTOLOGY_VERSION
    entities: list[ExtractedEntity] = Field(default_factory=list)
    valid_relationships: list[ExtractedRelationship] = Field(default_factory=list)
    hierarchy_nodes: list[HierarchyNode] = Field(default_factory=list)
    hierarchy_edges: list[HierarchyEdge] = Field(default_factory=list)
    cross_refs: list[CrossContractRef] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Extraction prompts
# ---------------------------------------------------------------------------

_ENTITY_PROMPT = """\
You are a legal contract entity extraction system specialized in the CUAD dataset.
Your task is to extract ONLY entities from the provided contract text.

RULES
1. Extract ONLY legally meaningful entities. Do NOT extract generic nouns.
2. Use ONLY these labels: Contract, Section, Clause, Party, Jurisdiction,
   EffectiveDate, ExpirationDate, RenewalTerm, LiabilityClause, IndemnityClause,
   PaymentTerm, ConfidentialityClause, TerminationClause, GoverningLawClause,
   Obligation, Risk, Amendment, ReferenceDocument
3. If unsure, use "Clause".
4. Every entity must have: entity_id, label, canonical_name, text_span, confidence.
5. canonical_name must be normalised and concise.
6. confidence between 0 and 1.

Return ONLY valid JSON. No explanations. No hallucinations.

OUTPUT FORMAT
{"entities": [{"entity_id": "party:acme_corp", "label": "Party",
  "canonical_name": "Acme Corp", "text_span": "Acme Corporation",
  "confidence": 0.98}]}\
"""

_RELATIONSHIP_PROMPT = """\
You are a legal contract relationship extraction system specialized in the CUAD dataset.
Extract semantic relationships between the provided entities.

RULES
1. Use ONLY these relationship types: SIGNED_BY, GOVERNED_BY, INDEMNIFIES,
   HAS_TERMINATION, HAS_RENEWAL, HAS_PAYMENT_TERM, REFERENCES, AMENDS,
   SUPERCEDES, OBLIGATES, LIMITS_LIABILITY, DISCLOSES_TO, HAS_CLAUSE
2. Only create relationships explicitly supported by the contract text.
3. Do NOT infer speculative relationships.
4. Every relationship: relationship_id, source_entity_id, target_entity_id,
   relationship_type, evidence_text, confidence.
5. source_entity_id and target_entity_id must reference entity_id values from the
   provided entity list.
6. evidence_text must contain the exact supporting text.

Return ONLY valid JSON. No explanations.

OUTPUT FORMAT
{"relationships": [{"relationship_id": "rel_001",
  "source_entity_id": "party:acme_corp", "target_entity_id": "party:beta_inc",
  "relationship_type": "INDEMNIFIES",
  "evidence_text": "Acme Corp shall indemnify Beta Inc",
  "confidence": 0.95}]}\
"""

_HIERARCHY_PROMPT = """\
You are a legal document structure extraction system.
Extract the hierarchical structure: Contract → Section → Clause → Chunk.

RELATIONSHIPS: HAS_SECTION, HAS_CLAUSE, HAS_CHUNK
Every node: node_id, node_type (Contract/Section/Clause/Chunk), title, sequence_number.
Every edge: source_id, target_id, relationship_type.
Preserve document ordering. Include sequence indexes.

Return ONLY valid JSON.
{"nodes": [...], "edges": [...]}\
"""

_CROSS_CONTRACT_PROMPT = """\
You are a legal contract lineage extraction system.
Identify references between contracts and legal documents.

RELATIONSHIP TYPES: REFERENCES, AMENDS, SUPERCEDES, REPLACES, ATTACHES,
  INCORPORATES_BY_REFERENCE

RULES
1. Extract ONLY explicit references. Do not infer.
2. Include referenced document names exactly as written.
3. Include supporting evidence text.

Return ONLY valid JSON.
{"references": [{"source_contract_id": "...",
  "target_document_name": "Master Services Agreement",
  "relationship_type": "AMENDS", "evidence_text": "...",
  "confidence": 0.93}]}\
"""

_VALIDATION_PROMPT = """\
You are a legal knowledge graph validation system.
Validate extracted relationships against the source contract text.

VALIDATION RULES
1. Remove relationships not supported by the evidence text.
2. Remove hallucinated entities.
3. Ensure ontology consistency (source/target types logically valid for rel type).
4. Verify confidence scores are accurate.

Return ONLY valid JSON.
{"valid_relationships": [...],
 "invalid_relationships": [{"relationship_id": "...", "reason": "..."}]}\
"""


# ---------------------------------------------------------------------------
# LLM agent factory (Ollama only)
# ---------------------------------------------------------------------------

def _make_agent(system_prompt: str) -> Agent:
    settings = load_settings()
    api_key  = settings.kg_llm_api_key  or settings.llm_api_key
    model_id = settings.kg_llm_model    or settings.llm_model
    base_url = settings.kg_llm_base_url or settings.llm_base_url
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    model    = OpenAIChatModel(model_id, provider=provider)
    return Agent(model, system_prompt=system_prompt)


def _clean_json(s: str) -> str:
    """Best-effort repair of common llama3.1 JSON output issues."""
    # Strip control characters (except \t \n \r which are valid in JSON strings)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", s)
    # Remove trailing commas before ] or } (JSON5 style — not valid JSON)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _parse_json(raw: str) -> dict:
    """Strip markdown fences, find and parse the first complete JSON object.

    Handles the four llama3.1 failure modes observed in production:
      1. Extra data  — multiple JSON objects in output (raw_decode stops at first)
      2. Control chars — raw_decode fails; strip and retry
      3. Trailing commas — JSON5-style `[...,]` / `{...,}` — strip and retry
      4. General malformed — return {} so the chunk is skipped gracefully
    """
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
    start = raw.find("{")
    if start == -1:
        return {}
    decoder = json.JSONDecoder()
    # Pass 1: raw_decode on original (handles extra-data case)
    try:
        obj, _ = decoder.raw_decode(raw, start)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    # Pass 2: clean (control chars + trailing commas) then raw_decode
    cleaned = _clean_json(raw[start:])
    try:
        obj, _ = decoder.raw_decode(cleaned)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


async def _run_agent(agent: Agent, prompt: str) -> dict:
    """Run agent, return parsed JSON dict. Returns {} on failure."""
    for attempt in range(4):
        try:
            result = await agent.run(prompt)
            return _parse_json(result.output)
        except Exception as exc:
            if "429" in str(exc) or "rate_limit" in str(exc).lower():
                wait = min(2 ** attempt, 30)
                logger.info("Rate limited, retrying in %ds", wait)
                await asyncio.sleep(wait)
                continue
            logger.warning("Agent call failed: %s", exc)
            return {}
    return {}


# ---------------------------------------------------------------------------
# Bronze store
# ---------------------------------------------------------------------------

class BronzeStore:
    """Saves/loads raw extraction artifacts from kg_raw_extractions JSONB table."""

    TABLE = "kg_raw_extractions"

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def initialize(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    contract_id      UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index      INT NOT NULL,
                    model_version    TEXT NOT NULL,
                    ontology_version TEXT NOT NULL DEFAULT '1.0',
                    raw_json         JSONB NOT NULL,
                    created_at       TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {self.TABLE}_dedup_idx
                ON {self.TABLE} (contract_id, chunk_index, model_version)
            """)

    async def save(self, artifact: BronzeArtifact) -> str:
        raw = json.dumps(artifact.model_dump())
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                INSERT INTO {self.TABLE}
                    (contract_id, chunk_index, model_version, ontology_version, raw_json)
                VALUES ($1::uuid, $2, $3, $4, $5::jsonb)
                ON CONFLICT (contract_id, chunk_index, model_version)
                DO UPDATE SET raw_json = EXCLUDED.raw_json,
                              ontology_version = EXCLUDED.ontology_version,
                              created_at = NOW()
                RETURNING id
                """,
                artifact.contract_id, artifact.chunk_index,
                artifact.model_version, artifact.ontology_version, raw,
            )
        return str(row["id"])

    async def load_for_contract(self, contract_id: str) -> list[BronzeArtifact]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT raw_json FROM {self.TABLE} WHERE contract_id = $1::uuid"
                " ORDER BY chunk_index",
                contract_id,
            )
        artifacts = []
        for row in rows:
            data = json.loads(row["raw_json"])
            try:
                artifacts.append(BronzeArtifact(**data))
            except Exception as exc:
                logger.warning("Skipping malformed bronze artifact: %s", exc)
        return artifacts

    async def clear_for_contract(self, contract_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self.TABLE} WHERE contract_id = $1::uuid", contract_id
            )


# ---------------------------------------------------------------------------
# Silver normalizer
# ---------------------------------------------------------------------------

class SilverNormalizer:
    """Reads Bronze artifacts, deduplicates, writes canonical tables."""

    def __init__(self, pool: asyncpg.Pool, confidence_threshold: float = 0.7) -> None:
        self._pool = pool
        self.threshold = confidence_threshold

    async def initialize(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS kg_staging_entities (
                    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    contract_id       UUID NOT NULL,
                    chunk_index       INT NOT NULL,
                    artifact_id       TEXT NOT NULL,
                    entity_id_raw     TEXT NOT NULL,
                    label             TEXT NOT NULL,
                    canonical_name    TEXT NOT NULL,
                    text_span         TEXT,
                    confidence        FLOAT NOT NULL,
                    created_at        TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS kg_staging_ent_contract_idx
                ON kg_staging_entities (contract_id)
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS kg_staging_relationships (
                    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    contract_id           UUID NOT NULL,
                    chunk_index           INT NOT NULL,
                    artifact_id           TEXT NOT NULL,
                    source_entity_id_raw  TEXT NOT NULL,
                    target_entity_id_raw  TEXT NOT NULL,
                    relationship_type     TEXT NOT NULL,
                    evidence_text         TEXT,
                    confidence            FLOAT NOT NULL,
                    validated             BOOLEAN DEFAULT TRUE,
                    created_at            TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS kg_staging_rel_contract_idx
                ON kg_staging_relationships (contract_id)
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS kg_canonical_entities (
                    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    contract_id          UUID NOT NULL,
                    label                TEXT NOT NULL,
                    canonical_name       TEXT NOT NULL,
                    confidence           FLOAT NOT NULL,
                    evidence_count       INT NOT NULL DEFAULT 1,
                    source_chunk_indices JSONB NOT NULL DEFAULT '[]',
                    created_at           TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE (contract_id, label, canonical_name)
                )
            """)
            # Migrate existing tables that predate source_chunk_indices
            await conn.execute("""
                ALTER TABLE kg_canonical_entities
                ADD COLUMN IF NOT EXISTS source_chunk_indices JSONB NOT NULL DEFAULT '[]'
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS kg_canonical_relationships (
                    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    contract_id      UUID NOT NULL,
                    source_entity_id UUID NOT NULL
                        REFERENCES kg_canonical_entities(id) ON DELETE CASCADE,
                    target_entity_id UUID NOT NULL
                        REFERENCES kg_canonical_entities(id) ON DELETE CASCADE,
                    relationship_type TEXT NOT NULL,
                    confidence        FLOAT NOT NULL,
                    evidence_texts    JSONB DEFAULT '[]',
                    created_at        TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE (contract_id, source_entity_id, target_entity_id,
                            relationship_type)
                )
            """)

    async def normalize(self, contract_id: str, artifacts: list[BronzeArtifact]) -> dict:
        """Deduplicate Bronze artifacts into canonical Silver tables."""
        await self.clear_for_contract(contract_id)

        # ---- stage raw entities & relationships ----
        async with self._pool.acquire() as conn:
            for art in artifacts:
                for ent in art.entities:
                    if ent.confidence < self.threshold:
                        continue
                    await conn.execute(
                        """
                        INSERT INTO kg_staging_entities
                            (contract_id, chunk_index, artifact_id, entity_id_raw,
                             label, canonical_name, text_span, confidence)
                        VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        contract_id, art.chunk_index, art.artifact_id,
                        ent.entity_id, ent.safe_label(), ent.canonical_name,
                        ent.text_span, ent.confidence,
                    )
                for rel in art.valid_relationships:
                    if rel.confidence < self.threshold or rel.safe_rel_type() is None:
                        continue
                    await conn.execute(
                        """
                        INSERT INTO kg_staging_relationships
                            (contract_id, chunk_index, artifact_id,
                             source_entity_id_raw, target_entity_id_raw,
                             relationship_type, evidence_text, confidence)
                        VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        contract_id, art.chunk_index, art.artifact_id,
                        rel.source_entity_id, rel.target_entity_id,
                        rel.safe_rel_type(), rel.evidence_text, rel.confidence,
                    )

        # ---- deduplicate entities → canonical ----
        # Key: (label, normalize(canonical_name)); keep highest confidence.
        async with self._pool.acquire() as conn:
            staging_ents = await conn.fetch(
                """
                SELECT DISTINCT ON (label, lower(regexp_replace(canonical_name, '\\s+', ' ', 'g')))
                    entity_id_raw, label, canonical_name, confidence
                FROM kg_staging_entities
                WHERE contract_id = $1::uuid
                ORDER BY label,
                         lower(regexp_replace(canonical_name, '\\s+', ' ', 'g')),
                         confidence DESC
                """,
                contract_id,
            )

            # raw_entity_id → canonical_entity_id mapping (for relationship resolution)
            raw_to_canonical: dict[str, str] = {}
            # Also map by (label, normalized_name) for cross-chunk resolution
            name_label_to_canonical: dict[tuple[str, str], str] = {}

            # Build chunk index map BEFORE the insert loop:
            # (label, normalized_name) → sorted list of chunk indices
            _ci_rows = await conn.fetch(
                "SELECT label, canonical_name, chunk_index"
                " FROM kg_staging_entities WHERE contract_id = $1::uuid",
                contract_id,
            )
            chunk_idx_map: dict[tuple[str, str], list[int]] = {}
            for r in _ci_rows:
                k = (r["label"], _normalize(r["canonical_name"]))
                chunk_idx_map.setdefault(k, []).append(r["chunk_index"])

            for row in staging_ents:
                key = (row["label"], _normalize(row["canonical_name"]))
                chunk_indices = json.dumps(sorted(set(chunk_idx_map.get(key, []))))
                canonical_id_row = await conn.fetchrow(
                    """
                    INSERT INTO kg_canonical_entities
                        (contract_id, label, canonical_name, confidence,
                         evidence_count, source_chunk_indices)
                    VALUES ($1::uuid, $2, $3, $4, 1, $5::jsonb)
                    ON CONFLICT (contract_id, label, canonical_name)
                    DO UPDATE SET
                        confidence           = GREATEST(kg_canonical_entities.confidence,
                                                        EXCLUDED.confidence),
                        evidence_count       = kg_canonical_entities.evidence_count + 1,
                        source_chunk_indices = (
                            SELECT jsonb_agg(DISTINCT val ORDER BY val)
                            FROM (
                                SELECT jsonb_array_elements_text(
                                    kg_canonical_entities.source_chunk_indices
                                    || EXCLUDED.source_chunk_indices
                                )::int AS val
                            ) t
                        )
                    RETURNING id
                    """,
                    contract_id, row["label"], row["canonical_name"],
                    row["confidence"], chunk_indices,
                )
                cid = str(canonical_id_row["id"])
                raw_to_canonical[row["entity_id_raw"]] = cid
                key = (row["label"], _normalize(row["canonical_name"]))
                name_label_to_canonical[key] = cid

            # Also build mapping from staging rows (many raw IDs → same canonical)
            all_staging = await conn.fetch(
                "SELECT entity_id_raw, label, canonical_name FROM kg_staging_entities"
                " WHERE contract_id = $1::uuid",
                contract_id,
            )
            for row in all_staging:
                key = (row["label"], _normalize(row["canonical_name"]))
                cid = name_label_to_canonical.get(key)
                if cid:
                    raw_to_canonical[row["entity_id_raw"]] = cid

            # ---- deduplicate relationships → canonical ----
            staging_rels = await conn.fetch(
                "SELECT source_entity_id_raw, target_entity_id_raw,"
                "       relationship_type, evidence_text, confidence"
                " FROM kg_staging_relationships WHERE contract_id = $1::uuid",
                contract_id,
            )

            canonical_rels: dict[tuple, dict] = {}
            for row in staging_rels:
                src_cid = raw_to_canonical.get(row["source_entity_id_raw"])
                tgt_cid = raw_to_canonical.get(row["target_entity_id_raw"])
                if not src_cid or not tgt_cid:
                    continue
                key = (src_cid, tgt_cid, row["relationship_type"])
                if key not in canonical_rels:
                    canonical_rels[key] = {
                        "confidence": row["confidence"],
                        "evidence_texts": [],
                    }
                cr = canonical_rels[key]
                cr["confidence"] = max(cr["confidence"], row["confidence"])
                if row["evidence_text"] and row["evidence_text"] not in cr["evidence_texts"]:
                    cr["evidence_texts"].append(row["evidence_text"])

            rel_count = 0
            for (src_cid, tgt_cid, rel_type), data in canonical_rels.items():
                await conn.execute(
                    """
                    INSERT INTO kg_canonical_relationships
                        (contract_id, source_entity_id, target_entity_id,
                         relationship_type, confidence, evidence_texts)
                    VALUES ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6::jsonb)
                    ON CONFLICT (contract_id, source_entity_id, target_entity_id,
                                 relationship_type)
                    DO UPDATE SET
                        confidence     = GREATEST(kg_canonical_relationships.confidence,
                                                  EXCLUDED.confidence),
                        evidence_texts = kg_canonical_relationships.evidence_texts ||
                                         EXCLUDED.evidence_texts
                    """,
                    contract_id, src_cid, tgt_cid, rel_type,
                    data["confidence"], json.dumps(data["evidence_texts"]),
                )
                rel_count += 1

        return {
            "canonical_entities": len(staging_ents),
            "canonical_relationships": rel_count,
        }

    async def clear_for_contract(self, contract_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM kg_staging_entities WHERE contract_id = $1::uuid", contract_id
            )
            await conn.execute(
                "DELETE FROM kg_staging_relationships WHERE contract_id = $1::uuid", contract_id
            )
            await conn.execute(
                "DELETE FROM kg_canonical_entities WHERE contract_id = $1::uuid", contract_id
            )


# ---------------------------------------------------------------------------
# Gold projector
# ---------------------------------------------------------------------------

class GoldProjector:
    """Projects canonical Silver entities/relationships into Apache AGE."""

    def __init__(self, pool: asyncpg.Pool, age_store: Any) -> None:
        self._pool = pool
        self._age = age_store

    async def project(self, contract_id: str) -> dict:
        async with self._pool.acquire() as conn:
            ent_rows = await conn.fetch(
                "SELECT id::text, label, canonical_name FROM kg_canonical_entities"
                " WHERE contract_id = $1::uuid",
                contract_id,
            )
            rel_rows = await conn.fetch(
                """
                SELECT r.relationship_type, r.confidence,
                       r.evidence_texts,
                       s.id::text AS src_pg_id, s.canonical_name AS src_name,
                       s.label   AS src_label,
                       t.id::text AS tgt_pg_id, t.canonical_name AS tgt_name,
                       t.label   AS tgt_label
                FROM kg_canonical_relationships r
                JOIN kg_canonical_entities s ON s.id = r.source_entity_id
                JOIN kg_canonical_entities t ON t.id = r.target_entity_id
                WHERE r.contract_id = $1::uuid
                """,
                contract_id,
            )

        pg_id_to_age_id: dict[str, str] = {}
        for row in ent_rows:
            age_id = await self._age.upsert_entity(
                name=row["canonical_name"],
                entity_type=row["label"],
                document_id=contract_id,
            )
            pg_id_to_age_id[row["id"]] = age_id

        rel_count = 0
        for row in rel_rows:
            src_age_id = pg_id_to_age_id.get(row["src_pg_id"])
            tgt_age_id = pg_id_to_age_id.get(row["tgt_pg_id"])
            if not src_age_id or not tgt_age_id:
                continue
            evidence = json.loads(row["evidence_texts"] or "[]")
            props = {"confidence": row["confidence"]}
            if evidence:
                props["evidence_text"] = evidence[0]
            await self._age.add_relationship(
                source_id=src_age_id,
                target_id=tgt_age_id,
                relationship_type=row["relationship_type"],
                document_id=contract_id,
                properties=props,
            )
            rel_count += 1

        return {
            "age_entities": len(ent_rows),
            "age_relationships": rel_count,
        }


# ---------------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------------

class ExtractionPipeline:
    """Orchestrates the 5-pass extraction, Bronze storage, Silver normalisation,
    and Gold projection for a single contract."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        age_store: Any,
        chunk_size: int = 1500,
        confidence_threshold: float = 0.7,
    ) -> None:
        self._pool = pool
        self._chunk_size = chunk_size
        self._threshold = confidence_threshold
        settings = load_settings()
        self._model_version = settings.kg_llm_model or settings.llm_model

        self._entity_agent       = _make_agent(_ENTITY_PROMPT)
        self._rel_agent          = _make_agent(_RELATIONSHIP_PROMPT)
        self._hierarchy_agent    = _make_agent(_HIERARCHY_PROMPT)
        self._cross_ref_agent    = _make_agent(_CROSS_CONTRACT_PROMPT)
        self._validation_agent   = _make_agent(_VALIDATION_PROMPT)

        self.bronze   = BronzeStore(pool)
        self.silver   = SilverNormalizer(pool, confidence_threshold)
        self.gold     = GoldProjector(pool, age_store)

    async def initialize(self) -> None:
        await self.bronze.initialize()
        await self.silver.initialize()

    def _chunk(self, text: str) -> list[str]:
        chunks = []
        for i in range(0, len(text), self._chunk_size):
            chunks.append(text[i : i + self._chunk_size])
        return chunks

    # ---- per-chunk passes ----

    async def _pass_entities(self, chunk: str) -> list[ExtractedEntity]:
        data = await _run_agent(self._entity_agent, f"Contract text:\n\n{chunk}")
        entities = []
        for raw in data.get("entities", []):
            try:
                ent = ExtractedEntity(**raw)
                if ent.confidence >= self._threshold:
                    entities.append(ent)
            except Exception:
                pass
        return entities

    async def _pass_relationships(
        self, chunk: str, entities: list[ExtractedEntity]
    ) -> list[ExtractedRelationship]:
        ent_json = json.dumps([e.model_dump() for e in entities], indent=2)
        prompt = f"Contract text:\n\n{chunk}\n\nExtracted entities:\n{ent_json}"
        data = await _run_agent(self._rel_agent, prompt)
        rels = []
        for raw in data.get("relationships", []):
            try:
                rel = ExtractedRelationship(**raw)
                if rel.confidence >= self._threshold and rel.safe_rel_type():
                    rels.append(rel)
            except Exception:
                pass
        return rels

    async def _pass_hierarchy(
        self, chunk: str
    ) -> tuple[list[HierarchyNode], list[HierarchyEdge]]:
        data = await _run_agent(self._hierarchy_agent, f"Contract text:\n\n{chunk}")
        nodes, edges = [], []
        for raw in data.get("nodes", []):
            try:
                nodes.append(HierarchyNode(**raw))
            except Exception:
                pass
        for raw in data.get("edges", []):
            try:
                edges.append(HierarchyEdge(**raw))
            except Exception:
                pass
        return nodes, edges

    async def _pass_cross_refs(
        self, chunk: str, contract_id: str
    ) -> list[CrossContractRef]:
        prompt = f"source_contract_id: {contract_id}\n\nContract text:\n\n{chunk}"
        data = await _run_agent(self._cross_ref_agent, prompt)
        refs = []
        for raw in data.get("references", []):
            try:
                ref = CrossContractRef(**raw)
                if ref.confidence >= self._threshold:
                    refs.append(ref)
            except Exception:
                pass
        return refs

    async def _pass_validate(
        self,
        chunk: str,
        entities: list[ExtractedEntity],
        rels: list[ExtractedRelationship],
    ) -> list[ExtractedRelationship]:
        ent_json = json.dumps([e.model_dump() for e in entities], indent=2)
        rel_json = json.dumps([r.model_dump() for r in rels], indent=2)
        prompt = (
            f"Contract text:\n\n{chunk}\n\n"
            f"Extracted entities:\n{ent_json}\n\n"
            f"Extracted relationships:\n{rel_json}"
        )
        data = await _run_agent(self._validation_agent, prompt)
        valid_ids = {
            r.get("relationship_id")
            for r in data.get("valid_relationships", [])
            if isinstance(r, dict)
        }
        if not valid_ids:
            # Validation returned nothing useful — keep all relationships.
            return rels
        return [r for r in rels if r.relationship_id in valid_ids]

    # ---- per-chunk orchestration ----

    async def _process_chunk(
        self, contract_id: str, chunk_index: int, chunk_text: str
    ) -> BronzeArtifact:
        entities    = await self._pass_entities(chunk_text)
        rels        = await self._pass_relationships(chunk_text, entities)
        h_nodes, h_edges = await self._pass_hierarchy(chunk_text)
        cross_refs  = await self._pass_cross_refs(chunk_text, contract_id)
        valid_rels  = await self._pass_validate(chunk_text, entities, rels)

        return BronzeArtifact(
            contract_id     = contract_id,
            chunk_index     = chunk_index,
            chunk_text      = chunk_text,
            model_version   = self._model_version,
            entities        = entities,
            valid_relationships = valid_rels,
            hierarchy_nodes = h_nodes,
            hierarchy_edges = h_edges,
            cross_refs      = cross_refs,
        )

    # ---- public API ----

    async def process_contract(
        self,
        contract_id: str,
        title: str,
        text: str,
    ) -> dict[str, Any]:
        """Full pipeline: extract → bronze → silver → gold."""
        chunks    = self._chunk(text)
        artifacts: list[BronzeArtifact] = []

        with Progress(
            TextColumn(f"[cyan]{title[:40]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("chunks", total=len(chunks))
            for i, chunk in enumerate(chunks):
                artifact = await self._process_chunk(contract_id, i, chunk)
                await self.bronze.save(artifact)
                artifacts.append(artifact)
                progress.advance(task)

        silver_stats = await self.silver.normalize(contract_id, artifacts)
        gold_stats   = await self.gold.project(contract_id)

        total_entities = sum(len(a.entities) for a in artifacts)
        total_rels     = sum(len(a.valid_relationships) for a in artifacts)

        return {
            "contract_id":  contract_id,
            "chunks":       len(chunks),
            "raw_entities": total_entities,
            "raw_relationships": total_rels,
            **silver_stats,
            **gold_stats,
        }

    async def project_contract(self, contract_id: str) -> dict[str, Any]:
        """Silver + Gold only — replay from existing Bronze (no LLM calls)."""
        artifacts = await self.bronze.load_for_contract(contract_id)
        if not artifacts:
            return {"error": f"No bronze artifacts found for contract {contract_id}"}
        silver_stats = await self.silver.normalize(contract_id, artifacts)
        gold_stats   = await self.gold.project(contract_id)
        return {**silver_stats, **gold_stats}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _fetch_contracts(
    limit: int | None = None,
    contract_id: str | None = None,
) -> list[dict[str, Any]]:
    settings = load_settings()
    conn = await asyncpg.connect(settings.database_url)
    try:
        if contract_id:
            rows = await conn.fetch(
                "SELECT id::text, title, content FROM documents"
                " WHERE id = $1::uuid AND content IS NOT NULL AND content <> ''",
                contract_id,
            )
        elif limit:
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
                " ORDER BY created_at",
            )
        return [{"id": r["id"], "title": r["title"], "content": r["content"]} for r in rows]
    finally:
        await conn.close()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bronze/Silver/Gold legal KG extraction pipeline"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all",         action="store_true", help="Process all contracts")
    mode.add_argument("--contract-id", metavar="UUID",      help="Process one contract")
    mode.add_argument("--project",     action="store_true",
                      help="Silver+Gold replay from Bronze (no LLM). "
                           "Use with --all or --contract-id.")
    parser.add_argument("--limit",    type=int, default=None)
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = load_settings()
    pool = await asyncpg.create_pool(
        settings.database_url, min_size=1, max_size=5, command_timeout=60
    )

    from rag.knowledge_graph import create_kg_store
    age_store = create_kg_store()
    await age_store.initialize()

    pipeline = ExtractionPipeline(
        pool=pool,
        age_store=age_store,
        chunk_size=settings.kg_extraction_chunk_size,
        confidence_threshold=settings.kg_confidence_threshold,
    )
    await pipeline.initialize()

    try:
        if args.project:
            contracts = await _fetch_contracts(
                limit=args.limit, contract_id=args.contract_id
            )
            console.print(f"[cyan]Replaying Silver+Gold for {len(contracts)} contract(s)…[/]")
            total = {"canonical_entities": 0, "canonical_relationships": 0,
                     "age_entities": 0, "age_relationships": 0}
            for c in contracts:
                stats = await pipeline.project_contract(c["id"])
                for k in total:
                    total[k] += stats.get(k, 0)
        else:
            contracts = await _fetch_contracts(
                limit=args.limit, contract_id=args.contract_id
            )
            console.print(f"[cyan]Processing {len(contracts)} contract(s)…[/]")
            console.print(
                f"[yellow]Model: {settings.kg_llm_model or settings.llm_model}  "
                f"via {settings.kg_llm_base_url or settings.llm_base_url}[/]"
            )
            await asyncio.sleep(3)

            total = {"chunks": 0, "raw_entities": 0, "raw_relationships": 0,
                     "canonical_entities": 0, "canonical_relationships": 0,
                     "age_entities": 0, "age_relationships": 0}
            for c in contracts:
                stats = await pipeline.process_contract(c["id"], c["title"], c["content"] or "")
                for k in total:
                    total[k] += stats.get(k, 0)

        table = Table(title="Pipeline Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        for k, v in total.items():
            table.add_row(k.replace("_", " ").title(), str(v))
        console.print(table)

    finally:
        await pool.close()
        await age_store.close()


if __name__ == "__main__":
    asyncio.run(main())
