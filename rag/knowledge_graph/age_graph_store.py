"""
Apache AGE knowledge graph store.

Module: rag.knowledge_graph.age_graph_store
============================================

Implements the same interface as PgGraphStore using Apache AGE — a PostgreSQL
extension that adds native openCypher graph queries.  Requires a running AGE
instance (see docker-compose.yml) and KG_BACKEND=age in .env.

How AGE works with asyncpg
--------------------------
AGE exposes Cypher through a SQL wrapper function::

    SELECT * FROM ag_catalog.cypher('graph_name', $$
        MATCH (e:Entity {entity_type: 'Party'})
        RETURN e.uuid, e.name
    $$) AS (uuid agtype, name agtype)

The ``agtype`` columns are returned as strings by asyncpg.  They look like
quoted JSON scalars (``"Acme Corp"``), so we strip the surrounding quotes.

Every connection must run two setup statements before issuing Cypher::

    LOAD 'age';
    SET search_path = ag_catalog, "$user", public;

We register this as an ``init`` callback on the asyncpg pool.

Vertex / edge model
-------------------
All vertices share the label ``Entity`` and carry ``entity_type`` as a property
(Party, Jurisdiction, LicenseClause, …).  This avoids the need to dynamically
compose Cypher with variable labels and makes full-graph queries simple.

Edge labels match relationship_type (PARTY_TO, GOVERNED_BY_LAW, …).

Switching from PgGraphStore
----------------------------
Change one line in .env::

    KG_BACKEND=age
    AGE_DATABASE_URL=postgresql://age_user:age_pass@localhost:5433/legal_graph

The factory function ``create_kg_store()`` in __init__.py returns the right
implementation automatically.

Usage
-----
    from rag.knowledge_graph.age_graph_store import AgeGraphStore

    store = AgeGraphStore()
    await store.initialize()

    eid = await store.upsert_entity("Acme Corp", "Party", doc_id)
    context = await store.search_as_context("governing law Delaware")

    await store.close()
"""

import asyncio
import json
import logging
import re
import uuid as _uuid
from typing import Any

import asyncpg

from rag.config.settings import load_settings
from rag.knowledge_graph.pg_graph_store import _normalize

logger = logging.getLogger(__name__)

_AGE_SETUP = [
    "LOAD 'age'",
    "SET search_path = ag_catalog, \"$user\", public",
]


def _unquote_agtype(value: str | None) -> str:
    """Strip surrounding double-quotes that AGE wraps around string agtypes."""
    if value is None:
        return ""
    s = str(value).strip()
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s


async def _age_init(conn: asyncpg.Connection) -> None:
    """Pool init callback: load AGE and set search_path for every new connection."""
    for stmt in _AGE_SETUP:
        await conn.execute(stmt)


class AgeGraphStore:
    """
    Apache AGE knowledge graph store.

    Implements the same public interface as PgGraphStore so the two are
    interchangeable via the ``create_kg_store()`` factory.
    """

    def __init__(self) -> None:
        self.settings = load_settings()
        self.pool: asyncpg.Pool | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

        age_url = self.settings.age_database_url
        if not age_url:
            raise ValueError(
                "AGE_DATABASE_URL is not set. Add it to .env or set kg_backend=postgres."
            )
        self._age_url = age_url
        self._graph = self.settings.age_graph_name

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the AGE graph and Entity vertex / edge structure."""
        async with self._init_lock:
            if self._initialized:
                return
            await self._do_initialize()

    async def _do_initialize(self) -> None:
        self.pool = await asyncpg.create_pool(
            self._age_url,
            min_size=1,
            max_size=5,
            command_timeout=60,
            init=_age_init,
        )
        async with self.pool.acquire() as conn:
            # Create the graph if it doesn't exist
            try:
                await conn.execute(
                    f"SELECT create_graph('{self._graph}')"
                )
                logger.info(f"Created AGE graph '{self._graph}'")
            except asyncpg.exceptions.InvalidSchemaNameError:
                logger.debug(f"AGE graph '{self._graph}' already exists")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.debug(f"AGE graph '{self._graph}' already exists")
                else:
                    raise

            # Create constraint index for deduplication on Entity vertices
            # AGE uses B-tree indexes on vertex properties
            try:
                await conn.execute(
                    f"SELECT * FROM ag_catalog.cypher('{self._graph}', $$"
                    " CREATE CONSTRAINT entity_dedup IF NOT EXISTS"
                    " FOR (e:Entity) REQUIRE (e.normalized_name, e.entity_type, e.document_id) IS UNIQUE"
                    "$$) AS (result agtype)"
                )
            except Exception:
                # Constraint creation syntax varies by AGE version — skip if unsupported
                pass

        self._initialized = True
        logger.info(f"AgeGraphStore initialized (graph='{self._graph}')")

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._initialized = False

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def _cypher(self, cypher_body: str) -> str:
        """Wrap Cypher in the AGE SQL function call."""
        return f"SELECT * FROM ag_catalog.cypher('{self._graph}', $${cypher_body}$$)"

    async def upsert_entity(
        self,
        name: str,
        entity_type: str,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """MERGE an Entity vertex; return its UUID property."""
        assert self.pool, "Call initialize() first"
        normalized = _normalize(name)
        entity_uuid = str(_uuid.uuid4())
        doc_id_str = document_id or ""
        meta_str = json.dumps(metadata or {}).replace("'", "\\'")
        name_escaped = name.replace("'", "\\'")
        meta_escaped = meta_str.replace('"', '\\"')

        cypher = (
            f"MERGE (e:Entity {{"
            f"normalized_name: '{normalized}', "
            f"entity_type: '{entity_type}', "
            f"document_id: '{doc_id_str}'"
            f"}}) "
            f"ON CREATE SET e.uuid = '{entity_uuid}', "
            f"e.name = '{name_escaped}', "
            f"e.metadata = '{meta_escaped}' "
            f"RETURN e.uuid"
        )

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                self._cypher(cypher) + " AS (uuid agtype)"
            )

        if rows:
            return _unquote_agtype(rows[0]["uuid"])
        return entity_uuid

    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        document_id: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Create a directed edge between two Entity vertices (by UUID)."""
        assert self.pool, "Call initialize() first"
        doc_id_str = document_id or ""
        props_str = json.dumps(properties or {}).replace("'", "\\'").replace('"', '\\"')
        rel_uuid = str(_uuid.uuid4())

        cypher = (
            f"MATCH (s:Entity {{uuid: '{source_id}'}}), "
            f"(t:Entity {{uuid: '{target_id}'}}) "
            f"CREATE (s)-[r:{relationship_type} {{"
            f"uuid: '{rel_uuid}', "
            f"document_id: '{doc_id_str}', "
            f"properties: '{props_str}'"
            f"}}]->(t) "
            f"RETURN r.uuid"
        )

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                self._cypher(cypher) + " AS (uuid agtype)"
            )

        if rows:
            return _unquote_agtype(rows[0]["uuid"])
        return rel_uuid

    async def clear_for_document(self, document_id: str) -> None:
        """Delete all Entity vertices (and edges) for a document."""
        assert self.pool, "Call initialize() first"
        cypher = (
            f"MATCH (e:Entity {{document_id: '{document_id}'}}) "
            f"DETACH DELETE e"
        )
        async with self.pool.acquire() as conn:
            await conn.execute(self._cypher(cypher) + " AS (r agtype)")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def search_entities(
        self,
        query: str,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Case-insensitive substring search over entity names."""
        assert self.pool, "Call initialize() first"
        query_esc = query.lower().replace("'", "\\'")

        if entity_type:
            cypher = (
                f"MATCH (e:Entity) "
                f"WHERE e.entity_type = '{entity_type}' "
                f"AND toLower(e.name) CONTAINS '{query_esc}' "
                f"RETURN e.uuid, e.name, e.entity_type, e.document_id "
                f"LIMIT {limit}"
            )
        else:
            cypher = (
                f"MATCH (e:Entity) "
                f"WHERE toLower(e.name) CONTAINS '{query_esc}' "
                f"RETURN e.uuid, e.name, e.entity_type, e.document_id "
                f"LIMIT {limit}"
            )

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                self._cypher(cypher)
                + " AS (uuid agtype, name agtype, entity_type agtype, document_id agtype)"
            )

        return [
            {
                "id": _unquote_agtype(r["uuid"]),
                "name": _unquote_agtype(r["name"]),
                "entity_type": _unquote_agtype(r["entity_type"]),
                "document_id": _unquote_agtype(r["document_id"]) or None,
                "document_title": None,  # would need a JOIN to documents table
            }
            for r in rows
        ]

    async def get_related_entities(
        self,
        entity_id: str,
        relationship_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return entities connected to entity_id (both directions)."""
        assert self.pool, "Call initialize() first"

        if relationship_type:
            cypher = (
                f"MATCH (s:Entity {{uuid: '{entity_id}'}})"
                f"-[r:{relationship_type}]-(t:Entity) "
                f"RETURN t.uuid, t.name, t.entity_type, type(r) AS rel_type "
                f"LIMIT {limit}"
            )
        else:
            cypher = (
                f"MATCH (s:Entity {{uuid: '{entity_id}'}})-[r]-(t:Entity) "
                f"RETURN t.uuid, t.name, t.entity_type, type(r) AS rel_type "
                f"LIMIT {limit}"
            )

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                self._cypher(cypher)
                + " AS (uuid agtype, name agtype, entity_type agtype, rel_type agtype)"
            )

        return [
            {
                "id": _unquote_agtype(r["uuid"]),
                "name": _unquote_agtype(r["name"]),
                "entity_type": _unquote_agtype(r["entity_type"]),
                "relationship_type": _unquote_agtype(r["rel_type"]),
            }
            for r in rows
        ]

    async def find_contracts_by_entity(
        self,
        entity_name: str,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return contract document_ids that contain the given named entity."""
        assert self.pool, "Call initialize() first"
        normalized = _normalize(entity_name)
        norm_esc = normalized.replace("'", "\\'")

        if entity_type:
            cypher = (
                f"MATCH (e:Entity {{normalized_name: '{norm_esc}', entity_type: '{entity_type}'}})"
                f"-[r]->(c:Entity {{entity_type: 'Contract'}}) "
                f"RETURN DISTINCT c.name, c.document_id "
                f"LIMIT {limit}"
            )
        else:
            cypher = (
                f"MATCH (e:Entity {{normalized_name: '{norm_esc}'}})"
                f"-[r]->(c:Entity {{entity_type: 'Contract'}}) "
                f"RETURN DISTINCT c.name, c.document_id "
                f"LIMIT {limit}"
            )

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                self._cypher(cypher) + " AS (title agtype, document_id agtype)"
            )

        return [
            {
                "title": _unquote_agtype(r["title"]),
                "document_id": _unquote_agtype(r["document_id"]),
            }
            for r in rows
        ]

    async def search_as_context(
        self,
        query: str,
        limit: int = 15,
    ) -> str:
        """Search entities + their relationships; return as LLM-ready context."""
        assert self.pool, "Call initialize() first"
        query_esc = query.lower().replace("'", "\\'")

        cypher = (
            f"MATCH (e:Entity)-[r]->(t:Entity) "
            f"WHERE toLower(e.name) CONTAINS '{query_esc}' "
            f"   OR toLower(t.name) CONTAINS '{query_esc}' "
            f"RETURN e.name, e.entity_type, type(r) AS rel, t.name, t.entity_type "
            f"LIMIT {limit}"
        )

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                self._cypher(cypher)
                + " AS (src_name agtype, src_type agtype, rel agtype,"
                  " tgt_name agtype, tgt_type agtype)"
            )

        if not rows:
            entities = await self.search_entities(query, limit=limit)
            if not entities:
                return "No relevant entities found in knowledge graph."
            lines = [f"- [{e['entity_type']}] {e['name']}" for e in entities]
            return "## Knowledge Graph — Entities\n" + "\n".join(lines)

        lines = [
            f"- [{_unquote_agtype(r['src_type'])}] {_unquote_agtype(r['src_name'])} "
            f"--{_unquote_agtype(r['rel'])}--> "
            f"[{_unquote_agtype(r['tgt_type'])}] {_unquote_agtype(r['tgt_name'])}"
            for r in rows
        ]
        return "## Knowledge Graph — Facts\n" + "\n".join(lines)

    async def get_graph_stats(self) -> dict[str, Any]:
        """Return vertex and edge counts broken down by type."""
        assert self.pool, "Call initialize() first"

        async with self.pool.acquire() as conn:
            vertex_rows = await conn.fetch(
                self._cypher(
                    "MATCH (e:Entity) RETURN e.entity_type, count(*) AS cnt"
                ) + " AS (entity_type agtype, cnt agtype)"
            )
            edge_rows = await conn.fetch(
                self._cypher(
                    "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS cnt"
                ) + " AS (rel_type agtype, cnt agtype)"
            )

        entities_by_type = {
            _unquote_agtype(r["entity_type"]): int(_unquote_agtype(r["cnt"]))
            for r in vertex_rows
        }
        rels_by_type = {
            _unquote_agtype(r["rel_type"]): int(_unquote_agtype(r["cnt"]))
            for r in edge_rows
        }

        return {
            "total_entities": sum(entities_by_type.values()),
            "total_relationships": sum(rels_by_type.values()),
            "entities_by_type": entities_by_type,
            "relationships_by_type": rels_by_type,
        }
