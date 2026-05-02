"""
PostgreSQL-backed knowledge graph store.

Module: rag.knowledge_graph.pg_graph_store
==========================================

Stores entities (nodes) and relationships (edges) for legal contract graphs
directly in PostgreSQL — no Neo4j required.

Schema
------
Two tables alongside ``documents`` and ``chunks``:

    kg_entities
        id UUID PK
        name TEXT            human-readable label ("Acme Corp", "Delaware law")
        entity_type TEXT     Party | Jurisdiction | Date | Clause | LicenseClause |
                             TerminationClause | RestrictionClause | IPClause | ...
        normalized_name TEXT lowercase stripped version for deduplication
        document_id UUID → documents(id) ON DELETE CASCADE
        metadata JSONB       extra fields (answer_start, clause_text, …)
        created_at TIMESTAMPTZ

    kg_relationships
        id UUID PK
        source_id UUID → kg_entities(id) ON DELETE CASCADE
        target_id UUID → kg_entities(id) ON DELETE CASCADE
        relationship_type TEXT  PARTY_TO | GOVERNED_BY | HAS_CLAUSE |
                                HAS_LICENSE_CLAUSE | HAS_TERMINATION_CLAUSE |
                                HAS_RESTRICTION | HAS_DATE | HAS_IP_CLAUSE | …
        document_id UUID → documents(id) ON DELETE CASCADE
        properties JSONB
        created_at TIMESTAMPTZ

Usage
-----
    from rag.knowledge_graph.pg_graph_store import PgGraphStore

    store = PgGraphStore()
    await store.initialize()

    # Add an entity
    eid = await store.upsert_entity("Acme Corp", "Party", doc_id)

    # Add a relationship (entity → contract)
    contract_eid = await store.upsert_entity(doc_title, "Contract", doc_id)
    await store.add_relationship(eid, contract_eid, "PARTY_TO", doc_id)

    # Search
    context = await store.search_as_context("Who are the parties to distributor agreements?")

    await store.close()
"""

import asyncio
import logging
import re
from typing import Any

import asyncpg

from rag.config.settings import load_settings

logger = logging.getLogger(__name__)


def _normalize(name: str) -> str:
    """Lowercase + collapse whitespace for deduplication."""
    return re.sub(r"\s+", " ", name.strip().lower())


class PgGraphStore:
    """PostgreSQL knowledge graph: entities + relationships in two tables."""

    def __init__(self) -> None:
        self.settings = load_settings()
        self.pool: asyncpg.Pool | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create tables and indexes if they do not exist."""
        async with self._init_lock:
            if self._initialized:
                return
            await self._do_initialize()

    async def _do_initialize(self) -> None:
        self.pool = await asyncpg.create_pool(
            self.settings.database_url,
            min_size=1,
            max_size=5,
            command_timeout=60,
        )
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS kg_entities (
                    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name             TEXT NOT NULL,
                    entity_type      TEXT NOT NULL,
                    normalized_name  TEXT NOT NULL,
                    document_id      UUID REFERENCES documents(id) ON DELETE CASCADE,
                    metadata         JSONB DEFAULT '{}',
                    created_at       TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            # Unique per (normalized_name, entity_type, document_id) — prevents
            # duplicate entities for the same contract.
            await conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS kg_entities_dedup_idx
                ON kg_entities (normalized_name, entity_type, COALESCE(document_id, '00000000-0000-0000-0000-000000000000'::uuid))
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS kg_entities_type_idx
                ON kg_entities (entity_type)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS kg_entities_name_idx
                ON kg_entities USING GIN (to_tsvector('english', name))
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS kg_relationships (
                    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source_id         UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
                    target_id         UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
                    relationship_type TEXT NOT NULL,
                    document_id       UUID REFERENCES documents(id) ON DELETE CASCADE,
                    properties        JSONB DEFAULT '{}',
                    created_at        TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS kg_rel_source_idx
                ON kg_relationships (source_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS kg_rel_target_idx
                ON kg_relationships (target_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS kg_rel_type_idx
                ON kg_relationships (relationship_type)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS kg_rel_document_idx
                ON kg_relationships (document_id)
            """)

        self._initialized = True
        logger.info("PgGraphStore initialized")

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._initialized = False

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert_entity(
        self,
        name: str,
        entity_type: str,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Insert or update an entity; return its UUID."""
        import json

        assert self.pool, "Call initialize() first"
        normalized = _normalize(name)
        meta_json = json.dumps(metadata or {})

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO kg_entities (name, entity_type, normalized_name, document_id, metadata)
                VALUES ($1, $2, $3, $4::uuid, $5::jsonb)
                ON CONFLICT (normalized_name, entity_type,
                    COALESCE(document_id, '00000000-0000-0000-0000-000000000000'::uuid))
                DO UPDATE SET
                    name     = EXCLUDED.name,
                    metadata = kg_entities.metadata || EXCLUDED.metadata
                RETURNING id
                """,
                name, entity_type, normalized, document_id, meta_json,
            )
        return str(row["id"])

    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        document_id: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Add a directed edge; return its UUID."""
        import json

        assert self.pool, "Call initialize() first"
        props_json = json.dumps(properties or {})

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO kg_relationships
                    (source_id, target_id, relationship_type, document_id, properties)
                VALUES ($1::uuid, $2::uuid, $3, $4::uuid, $5::jsonb)
                RETURNING id
                """,
                source_id, target_id, relationship_type, document_id, props_json,
            )
        return str(row["id"])

    async def clear_for_document(self, document_id: str) -> None:
        """Delete all entities (and cascading relationships) for a document."""
        assert self.pool, "Call initialize() first"
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM kg_entities WHERE document_id = $1::uuid", document_id
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def search_entities(
        self,
        query: str,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Full-text search over entity names."""
        assert self.pool, "Call initialize() first"
        async with self.pool.acquire() as conn:
            if entity_type:
                rows = await conn.fetch(
                    """
                    SELECT e.id, e.name, e.entity_type, e.normalized_name,
                           e.document_id, e.metadata,
                           d.title AS document_title, d.source AS document_source
                    FROM kg_entities e
                    LEFT JOIN documents d ON d.id = e.document_id
                    WHERE e.entity_type = $1
                      AND to_tsvector('english', e.name) @@ plainto_tsquery('english', $2)
                    ORDER BY ts_rank(to_tsvector('english', e.name),
                                     plainto_tsquery('english', $2)) DESC
                    LIMIT $3
                    """,
                    entity_type, query, limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT e.id, e.name, e.entity_type, e.normalized_name,
                           e.document_id, e.metadata,
                           d.title AS document_title, d.source AS document_source
                    FROM kg_entities e
                    LEFT JOIN documents d ON d.id = e.document_id
                    WHERE to_tsvector('english', e.name) @@ plainto_tsquery('english', $1)
                    ORDER BY ts_rank(to_tsvector('english', e.name),
                                     plainto_tsquery('english', $1)) DESC
                    LIMIT $2
                    """,
                    query, limit,
                )
        return [dict(r) for r in rows]

    async def get_related_entities(
        self,
        entity_id: str,
        relationship_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return entities connected to entity_id (both directions)."""
        assert self.pool, "Call initialize() first"
        async with self.pool.acquire() as conn:
            if relationship_type:
                rows = await conn.fetch(
                    """
                    SELECT e.id, e.name, e.entity_type, r.relationship_type,
                           'outgoing' AS direction,
                           d.title AS document_title
                    FROM kg_relationships r
                    JOIN kg_entities e ON e.id = r.target_id
                    LEFT JOIN documents d ON d.id = r.document_id
                    WHERE r.source_id = $1::uuid AND r.relationship_type = $2
                    UNION ALL
                    SELECT e.id, e.name, e.entity_type, r.relationship_type,
                           'incoming' AS direction,
                           d.title AS document_title
                    FROM kg_relationships r
                    JOIN kg_entities e ON e.id = r.source_id
                    LEFT JOIN documents d ON d.id = r.document_id
                    WHERE r.target_id = $1::uuid AND r.relationship_type = $2
                    LIMIT $3
                    """,
                    entity_id, relationship_type, limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT e.id, e.name, e.entity_type, r.relationship_type,
                           'outgoing' AS direction,
                           d.title AS document_title
                    FROM kg_relationships r
                    JOIN kg_entities e ON e.id = r.target_id
                    LEFT JOIN documents d ON d.id = r.document_id
                    WHERE r.source_id = $1::uuid
                    UNION ALL
                    SELECT e.id, e.name, e.entity_type, r.relationship_type,
                           'incoming' AS direction,
                           d.title AS document_title
                    FROM kg_relationships r
                    JOIN kg_entities e ON e.id = r.source_id
                    LEFT JOIN documents d ON d.id = r.document_id
                    WHERE r.target_id = $1::uuid
                    LIMIT $2
                    """,
                    entity_id, limit,
                )
        return [dict(r) for r in rows]

    async def find_contracts_by_entity(
        self,
        entity_name: str,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return contracts (documents) that contain the given named entity."""
        assert self.pool, "Call initialize() first"
        normalized = _normalize(entity_name)
        async with self.pool.acquire() as conn:
            if entity_type:
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT d.id, d.title, d.source,
                           e.entity_type, e.name AS matched_entity
                    FROM kg_entities e
                    JOIN documents d ON d.id = e.document_id
                    WHERE e.normalized_name = $1 AND e.entity_type = $2
                    ORDER BY d.title
                    LIMIT $3
                    """,
                    normalized, entity_type, limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT d.id, d.title, d.source,
                           e.entity_type, e.name AS matched_entity
                    FROM kg_entities e
                    JOIN documents d ON d.id = e.document_id
                    WHERE e.normalized_name = $1
                    ORDER BY d.title
                    LIMIT $2
                    """,
                    normalized, limit,
                )
        return [dict(r) for r in rows]

    async def get_contract_graph(
        self,
        document_id: str,
    ) -> dict[str, Any]:
        """Return all entities and relationships for one contract."""
        assert self.pool, "Call initialize() first"
        async with self.pool.acquire() as conn:
            entities = await conn.fetch(
                """
                SELECT id, name, entity_type, metadata
                FROM kg_entities
                WHERE document_id = $1::uuid
                ORDER BY entity_type, name
                """,
                document_id,
            )
            rels = await conn.fetch(
                """
                SELECT r.relationship_type,
                       s.name AS source_name, s.entity_type AS source_type,
                       t.name AS target_name, t.entity_type AS target_type
                FROM kg_relationships r
                JOIN kg_entities s ON s.id = r.source_id
                JOIN kg_entities t ON t.id = r.target_id
                WHERE r.document_id = $1::uuid
                """,
                document_id,
            )
        return {
            "entities": [dict(e) for e in entities],
            "relationships": [dict(r) for r in rels],
        }

    async def search_as_context(
        self,
        query: str,
        limit: int = 15,
    ) -> str:
        """Search entities and their relationships; return as LLM-ready context."""
        assert self.pool, "Call initialize() first"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    e.name            AS entity_name,
                    e.entity_type,
                    r.relationship_type,
                    t.name            AS related_name,
                    t.entity_type     AS related_type,
                    d.title           AS document_title
                FROM kg_entities e
                JOIN kg_relationships r ON r.source_id = e.id
                JOIN kg_entities t      ON t.id = r.target_id
                LEFT JOIN documents d   ON d.id = r.document_id
                WHERE to_tsvector('english', e.name || ' ' || COALESCE(t.name,''))
                      @@ plainto_tsquery('english', $1)
                ORDER BY e.entity_type, r.relationship_type
                LIMIT $2
                """,
                query, limit,
            )

        if not rows:
            # Fall back to entity-only search
            entities = await self.search_entities(query, limit=limit)
            if not entities:
                return "No relevant entities found in knowledge graph."
            lines = [
                f"- [{e['entity_type']}] {e['name']}"
                + (f" (from: {e['document_title']})" if e.get("document_title") else "")
                for e in entities
            ]
            return "## Knowledge Graph — Entities\n" + "\n".join(lines)

        lines: list[str] = []
        for r in rows:
            doc = f" (contract: {r['document_title']})" if r["document_title"] else ""
            lines.append(
                f"- [{r['entity_type']}] {r['entity_name']} "
                f"--{r['relationship_type']}--> "
                f"[{r['related_type']}] {r['related_name']}{doc}"
            )
        return "## Knowledge Graph — Facts\n" + "\n".join(lines)

    async def get_graph_stats(self) -> dict[str, Any]:
        """Return entity and relationship counts broken down by type."""
        assert self.pool, "Call initialize() first"
        async with self.pool.acquire() as conn:
            total_entities = await conn.fetchval("SELECT COUNT(*) FROM kg_entities")
            total_rels = await conn.fetchval("SELECT COUNT(*) FROM kg_relationships")
            entity_counts = await conn.fetch(
                "SELECT entity_type, COUNT(*) AS cnt FROM kg_entities GROUP BY 1 ORDER BY cnt DESC"
            )
            rel_counts = await conn.fetch(
                "SELECT relationship_type, COUNT(*) AS cnt FROM kg_relationships GROUP BY 1 ORDER BY cnt DESC"
            )
        return {
            "total_entities": total_entities,
            "total_relationships": total_rels,
            "entities_by_type": {r["entity_type"]: r["cnt"] for r in entity_counts},
            "relationships_by_type": {r["relationship_type"]: r["cnt"] for r in rel_counts},
        }

    async def run_cypher_query(self, cypher: str) -> str:
        """Cypher is not supported on the PgGraphStore SQL backend.

        Set KG_BACKEND=age and start the Apache AGE container to use Cypher queries.
        """
        return (
            "Cypher queries require the Apache AGE backend (KG_BACKEND=age). "
            "The current backend is PgGraphStore (SQL tables). "
            "Start the AGE container and set KG_BACKEND=age in .env to enable this tool."
        )
