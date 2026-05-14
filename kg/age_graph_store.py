"""
Apache AGE knowledge graph store.

Module: kg.age_graph_store
============================================

Apache AGE is a PostgreSQL extension that adds native openCypher graph queries.
Requires a running AGE instance (see docker-compose.yml).

How AGE works with asyncpg
--------------------------
AGE exposes Cypher through a SQL wrapper function::

    SELECT * FROM ag_catalog.cypher('graph_name', $$
        MATCH (e:Party) RETURN e.uuid, e.name
    $$) AS (uuid agtype, name agtype)

The ``agtype`` columns are returned as strings by asyncpg.  They look like
quoted JSON scalars (``"Acme Corp"``), so we strip the surrounding quotes.

Every connection must run two setup statements before issuing Cypher::

    LOAD 'age';
    SET search_path = ag_catalog, "$user", public;

We register this as an ``init`` callback on the asyncpg pool.

Usage
-----
    from kg.age_graph_store import AgeGraphStore

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
from contextlib import asynccontextmanager
from typing import Any

import asyncpg

from rag.config.settings import load_settings
from kg.constants import VALID_LABELS as _VALID_LABELS
from kg.constants import VALID_REL_TYPES as _VALID_REL_TYPES
from kg.entity_index import EntityIndex


def _normalize(name: str) -> str:
    """Lowercase + collapse whitespace for deduplication."""
    return re.sub(r"\s+", " ", name.strip().lower())


def _safe_label(entity_type: str) -> str:
    """Return entity_type if it is a known vertex label, else 'Clause'."""
    cleaned = re.sub(r"[^A-Za-z]", "", entity_type)
    return cleaned if cleaned in _VALID_LABELS else "Clause"


def _safe_rel_type(rel_type: str) -> str | None:
    """Return rel_type if valid, else None (caller should skip the edge)."""
    cleaned = re.sub(r"[^A-Z0-9_]", "", rel_type.upper())
    return cleaned if cleaned in _VALID_REL_TYPES else None


def _parse_return_aliases(cypher: str) -> list[str]:
    """Extract display names from the RETURN clause for building the AGE AS list."""
    cypher = cypher.strip().rstrip(";")
    m = re.search(
        r"\bRETURN\b\s+(.*?)(?:\s+(?:ORDER\s+BY|LIMIT|SKIP|UNION)\b|$)",
        cypher,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return ["c0"]
    body = m.group(1).strip()
    # Split on commas that are not inside parentheses
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in body:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    aliases: list[str] = []
    for i, part in enumerate(parts):
        part = part.strip()
        alias_m = re.search(r"\bAS\s+(\w+)\s*$", part, re.IGNORECASE)
        if alias_m:
            aliases.append(alias_m.group(1))
        else:
            toks = re.findall(r"\w+", part)
            aliases.append(toks[-1] if toks else f"c{i}")
    return aliases or ["c0"]


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
    """Apache AGE knowledge graph store."""

    def __init__(self) -> None:
        self.settings = load_settings()
        self.pool: asyncpg.Pool | None = None
        self._entity_index: EntityIndex | None = None
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
        async with self._conn() as conn:
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

        # Spin up the entity shadow index (tsvector + pgvector in main DB).
        # Graceful degradation: if DATABASE_URL is absent or the main DB is
        # unreachable we fall back to the O(n) AGE CONTAINS scan.
        try:
            self._entity_index = EntityIndex()
            await self._entity_index.initialize()
        except Exception as exc:
            logger.warning(
                "EntityIndex init failed (%s) — search_entities will use CONTAINS scan",
                exc,
            )
            self._entity_index = None

        self._initialized = True
        logger.info(f"AgeGraphStore initialized (graph='{self._graph}')")

    async def close(self) -> None:
        if self._entity_index:
            await self._entity_index.close()
            self._entity_index = None
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._initialized = False

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _conn(self):
        """Acquire a pool connection with AGE loaded and search_path set.

        asyncpg resets connection state (RESET ALL) when connections are
        returned to the pool, which clears both ``LOAD 'age'`` and the
        custom ``search_path``.  This context manager re-applies both on
        every acquire so callers don't need to think about it.
        """
        async with self.pool.acquire() as conn:
            for stmt in _AGE_SETUP:
                await conn.execute(stmt)
            yield conn

    def _cypher(self, cypher_body: str) -> str:
        """Wrap Cypher in the AGE SQL function call."""
        logger.debug("CYPHER: %s", cypher_body)
        return f"SELECT * FROM ag_catalog.cypher('{self._graph}', $${cypher_body}$$)"

    async def upsert_entity(
        self,
        name: str,
        entity_type: str,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """MERGE a typed vertex (e.g. :Party, :Contract); return its UUID property.

        entity_type becomes the Cypher vertex label directly — validated against
        _VALID_LABELS to prevent injection.  Falls back to :Clause for unknowns.
        """
        assert self.pool, "Call initialize() first"
        label      = _safe_label(entity_type)
        normalized = _normalize(name)
        entity_uuid = str(_uuid.uuid4())
        doc_id_str  = document_id or ""
        name_esc  = name.replace("\\", "\\\\").replace('"', '\\"')
        meta_esc  = json.dumps(metadata or {}).replace("\\", "\\\\").replace('"', '\\"')
        norm_esc  = normalized.replace("\\", "\\\\").replace('"', '\\"')

        # Distinct label per entity type — label IS the type, no entity_type property.
        # e.label stored as a property for label-agnostic traversals (MATCH (n) …).
        # AGE 1.7: no ON CREATE SET, so use COALESCE to preserve existing uuid.
        cypher = (
            f'MERGE (e:{label} {{'
            f'normalized_name: "{norm_esc}", '
            f'document_id: "{doc_id_str}"'
            f'}}) '
            f'SET e.uuid = COALESCE(e.uuid, "{entity_uuid}"), '
            f'e.name = "{name_esc}", '
            f'e.label = "{label}", '
            f'e.metadata = "{meta_esc}" '
            f'RETURN e.uuid'
        )

        async with self._conn() as conn:
            rows = await conn.fetch(
                self._cypher(cypher) + " AS (uuid agtype)"
            )

        result_uuid = _unquote_agtype(rows[0]["uuid"]) if rows else entity_uuid

        if self._entity_index:
            try:
                await self._entity_index.upsert(
                    result_uuid, name, label, doc_id_str
                )
            except Exception as exc:
                logger.debug("EntityIndex.upsert failed for %r: %s", name, exc)

        return result_uuid

    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        document_id: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Create a directed edge between two vertices (matched by uuid property)."""
        assert self.pool, "Call initialize() first"
        safe_type = _safe_rel_type(relationship_type)
        if safe_type is None:
            logger.warning("Skipping unknown relationship type %r", relationship_type)
            return ""
        doc_id_str = document_id or ""
        props_esc  = json.dumps(properties or {}).replace("\\", "\\\\").replace('"', '\\"')
        rel_uuid   = str(_uuid.uuid4())

        # MATCH by uuid property — label-agnostic so it works with distinct labels.
        cypher = (
            f'MATCH (s {{uuid: "{source_id}"}}), '
            f'(t {{uuid: "{target_id}"}}) '
            f'CREATE (s)-[r:{safe_type} {{'
            f'uuid: "{rel_uuid}", '
            f'document_id: "{doc_id_str}", '
            f'properties: "{props_esc}"'
            f'}}]->(t) '
            f'RETURN r.uuid'
        )

        async with self._conn() as conn:
            rows = await conn.fetch(
                self._cypher(cypher) + " AS (uuid agtype)"
            )

        if rows:
            return _unquote_agtype(rows[0]["uuid"])
        return rel_uuid

    async def clear_for_document(self, document_id: str) -> None:
        """Delete all Entity vertices (and edges) for a document."""
        assert self.pool, "Call initialize() first"
        doc_id_esc = document_id.replace("\\", "\\\\").replace('"', '\\"')
        cypher = (
            f'MATCH (e:Entity {{document_id: "{doc_id_esc}"}}) '
            f'DETACH DELETE e'
        )
        async with self._conn() as conn:
            await conn.execute(self._cypher(cypher) + " AS (r agtype)")
        if self._entity_index:
            await self._entity_index.delete_for_document(document_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def search_entities(
        self,
        query: str,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Hybrid tsvector + vector search via EntityIndex; falls back to CONTAINS scan."""
        assert self.pool, "Call initialize() first"

        if self._entity_index and self._entity_index._initialized:
            label_filter = _safe_label(entity_type) if entity_type else None
            rows = await self._entity_index.hybrid_search(query, label=label_filter, limit=limit)
            return [
                {
                    "id": r["age_uuid"],
                    "name": r["name"],
                    "entity_type": r["label"],
                    "document_id": r["document_id"] or None,
                    "document_title": None,
                }
                for r in rows
            ]

        # --- fallback: O(n) CONTAINS scan in AGE ---
        query_esc = query.lower().replace("\\", "\\\\").replace('"', '\\"')

        if entity_type:
            # Use the validated label as a Cypher label filter for performance.
            label = _safe_label(entity_type)
            cypher = (
                f'MATCH (e:{label}) '
                f'WHERE toLower(e.name) CONTAINS "{query_esc}" '
                f'RETURN e.uuid, e.name, e.label, e.document_id '
                f'LIMIT {limit}'
            )
        else:
            # No label filter — scans all vertex tables.
            cypher = (
                f'MATCH (e) '
                f'WHERE toLower(e.name) CONTAINS "{query_esc}" '
                f'RETURN e.uuid, e.name, e.label, e.document_id '
                f'LIMIT {limit}'
            )

        async with self._conn() as conn:
            rows = await conn.fetch(
                self._cypher(cypher)
                + " AS (uuid agtype, name agtype, label agtype, document_id agtype)"
            )

        return [
            {
                "id": _unquote_agtype(r["uuid"]),
                "name": _unquote_agtype(r["name"]),
                "entity_type": _unquote_agtype(r["label"]),
                "document_id": _unquote_agtype(r["document_id"]) or None,
                "document_title": None,
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

        eid_esc = entity_id.replace("\\", "\\\\").replace('"', '\\"')
        if relationship_type:
            safe_type = _safe_rel_type(relationship_type) or "HAS_CLAUSE"
            cypher = (
                f'MATCH (s {{uuid: "{eid_esc}"}})'
                f'-[r:{safe_type}]-(t) '
                f'RETURN t.uuid, t.name, t.label, type(r) AS rel_type '
                f'LIMIT {limit}'
            )
        else:
            cypher = (
                f'MATCH (s {{uuid: "{eid_esc}"}})-[r]-(t) '
                f'RETURN t.uuid, t.name, t.label, type(r) AS rel_type '
                f'LIMIT {limit}'
            )

        async with self._conn() as conn:
            rows = await conn.fetch(
                self._cypher(cypher)
                + " AS (uuid agtype, name agtype, label agtype, rel_type agtype)"
            )

        return [
            {
                "id": _unquote_agtype(r["uuid"]),
                "name": _unquote_agtype(r["name"]),
                "entity_type": _unquote_agtype(r["label"]),
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
        norm_esc = normalized.replace("\\", "\\\\").replace('"', '\\"')

        if entity_type:
            label = _safe_label(entity_type)
            cypher = (
                f'MATCH (e:{label} {{normalized_name: "{norm_esc}"}})'
                f'-[r]->(c:Contract) '
                f'RETURN DISTINCT c.name, c.document_id '
                f'LIMIT {limit}'
            )
        else:
            cypher = (
                f'MATCH (e {{normalized_name: "{norm_esc}"}})'
                f'-[r]->(c:Contract) '
                f'RETURN DISTINCT c.name, c.document_id '
                f'LIMIT {limit}'
            )

        async with self._conn() as conn:
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
        query_esc = query.lower().replace("\\", "\\\\").replace('"', '\\"')

        cypher = (
            f'MATCH (e)-[r]->(t) '
            f'WHERE toLower(e.name) CONTAINS "{query_esc}" '
            f'   OR toLower(t.name) CONTAINS "{query_esc}" '
            f'RETURN e.name, e.label, type(r) AS rel, t.name, t.label '
            f'LIMIT {limit}'
        )

        async with self._conn() as conn:
            rows = await conn.fetch(
                self._cypher(cypher)
                + " AS (src_name agtype, src_label agtype, rel agtype,"
                  " tgt_name agtype, tgt_label agtype)"
            )

        if not rows:
            entities = await self.search_entities(query, limit=limit)
            if not entities:
                return "No relevant entities found in knowledge graph."
            lines = [f"- [{e['entity_type']}] {e['name']}" for e in entities]
            return "## Knowledge Graph — Entities\n" + "\n".join(lines)

        lines = [
            f"- [{_unquote_agtype(r['src_label'])}] {_unquote_agtype(r['src_name'])} "
            f"--{_unquote_agtype(r['rel'])}--> "
            f"[{_unquote_agtype(r['tgt_label'])}] {_unquote_agtype(r['tgt_name'])}"
            for r in rows
        ]
        return "## Knowledge Graph — Facts\n" + "\n".join(lines)

    async def get_graph_stats(self) -> dict[str, Any]:
        """Return vertex and edge counts broken down by type."""
        assert self.pool, "Call initialize() first"

        async with self._conn() as conn:
            vertex_rows = await conn.fetch(
                self._cypher(
                    "MATCH (e) RETURN e.label, count(*) AS cnt"
                ) + " AS (label agtype, cnt agtype)"
            )
            edge_rows = await conn.fetch(
                self._cypher(
                    "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS cnt"
                ) + " AS (rel_type agtype, cnt agtype)"
            )

        entities_by_type = {
            _unquote_agtype(r["label"]): int(_unquote_agtype(r["cnt"]))
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

    async def run_cypher_query(self, cypher: str) -> str:
        """Execute a read-only Cypher MATCH query and return results as a table string.

        Only MATCH/RETURN queries are permitted; CREATE/MERGE/SET/DELETE/DROP/DETACH
        are blocked.  Column count is inferred from the RETURN clause so callers do
        not need to supply an explicit AS list.
        """
        assert self.pool, "Call initialize() first"

        if re.search(
            r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH)\b", cypher, re.IGNORECASE
        ):
            return "Error: only read-only MATCH queries are permitted."

        aliases = _parse_return_aliases(cypher)
        as_clause = ", ".join(f"c{i} agtype" for i in range(len(aliases)))

        async with self._conn() as conn:
            try:
                rows = await conn.fetch(
                    self._cypher(cypher) + f" AS ({as_clause})"
                )
            except Exception as exc:
                return f"Cypher error: {exc}"

        if not rows:
            return "No results."

        header = " | ".join(aliases)
        sep = "-" * max(len(header), 10)
        lines = [header, sep]
        for row in rows:
            vals = [_unquote_agtype(row[f"c{i}"]) for i in range(len(aliases))]
            lines.append(" | ".join(vals))
        n = len(rows)
        lines.append(f"\n({n} row{'s' if n != 1 else ''})")
        return "\n".join(lines)
