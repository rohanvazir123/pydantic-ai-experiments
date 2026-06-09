"""Apache AGE knowledge graph store.

AGE exposes openCypher through a SQL wrapper function:
    SELECT * FROM ag_catalog.cypher('graph_name', $$ MATCH … $$) AS (col agtype)

Every connection must run two setup statements before issuing Cypher:
    LOAD 'age';
    SET search_path = ag_catalog, "$user", public;

These are registered as an asyncpg pool init callback AND re-applied on every
pool.acquire() because AGE state is reset by RESET ALL on connection return.

Graph naming convention: {prefix}_{tenant_id}_{corpus_id}
e.g. kg_acme_corp_hr_policies

IMPORTANT: Do NOT use CypherExporter from docling-graph.
CypherExporter generates Neo4j-compatible raw CREATE statements that are
incompatible with AGE's ag_catalog.cypher() SQL wrapper syntax.
Instead, iterate PipelineContext.knowledge_graph (NetworkX DiGraph) directly
and call _upsert_vertex() + _add_edge() per node/edge.
"""

import logging
import re
import uuid as _uuid
from contextlib import asynccontextmanager
from typing import Any, TYPE_CHECKING

import asyncpg

from knowledge.config.settings import Settings, load_settings

if TYPE_CHECKING:
    from networkx import DiGraph

logger = logging.getLogger(__name__)

_AGE_SETUP = [
    "LOAD 'age'",
    'SET search_path = ag_catalog, "$user", public',
]


def _unquote_agtype(value: str | None) -> str:
    """Strip surrounding double-quotes AGE wraps around string agtypes."""
    if value is None:
        return ""
    s = str(value).strip()
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s


def _sanitize_label(label: str) -> str:
    """Strip non-alphanumeric; capitalise first letter; default 'Entity'."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", label)
    if not cleaned:
        return "Entity"
    return cleaned[0].upper() + cleaned[1:]


def _sanitize_rel_type(rel_type: str) -> str:
    """Uppercase; strip non-alphanumeric except underscore; default 'RELATED_TO'."""
    cleaned = re.sub(r"[^A-Z0-9_]", "", rel_type.upper())
    return cleaned or "RELATED_TO"


def _parse_return_aliases(cypher: str) -> list[str]:
    """Infer column aliases from the RETURN clause for the AGE AS list."""
    cypher = cypher.strip().rstrip(";")
    m = re.search(
        r"\bRETURN\b\s+(.*?)(?:\s+(?:ORDER\s+BY|LIMIT|SKIP|UNION)\b|$)",
        cypher, re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return ["c0"]
    body = m.group(1).strip()
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
        alias_m = re.search(r"\bAS\s+(\w+)\s*$", part, re.IGNORECASE)
        if alias_m:
            aliases.append(alias_m.group(1))
        else:
            toks = re.findall(r"\w+", part)
            aliases.append(toks[-1] if toks else f"c{i}")
    return aliases or ["c0"]


async def _age_init(conn: asyncpg.Connection) -> None:
    for stmt in _AGE_SETUP:
        await conn.execute(stmt)


class AgeGraphStore:
    """Apache AGE knowledge graph store for per-corpus openCypher graphs."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_settings()
        self._pool: asyncpg.Pool | None = None

    def _graph_name(self, tenant_id: str, corpus_id: str) -> str:
        prefix = self._settings.age_graph_prefix
        safe_t = re.sub(r"[^A-Za-z0-9]", "_", tenant_id)
        safe_c = re.sub(r"[^A-Za-z0-9]", "_", corpus_id)
        return f"{prefix}_{safe_t}_{safe_c}"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._settings.age_database_url,
            min_size=1,
            max_size=5,
            command_timeout=self._settings.db_query_timeout_s,
            init=_age_init,
        )
        logger.info("AgeGraphStore initialised (age_database_url set)")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def _conn(self):
        """Acquire a connection with AGE loaded and search_path set."""
        assert self._pool, "Call initialize() first"
        async with self._pool.acquire() as conn:
            for stmt in _AGE_SETUP:
                await conn.execute(stmt)
            yield conn

    def _cypher_sql(self, graph_name: str, cypher: str, as_clause: str) -> str:
        return f"SELECT * FROM ag_catalog.cypher('{graph_name}', $${cypher}$$) AS ({as_clause})"

    # ── Graph lifecycle ───────────────────────────────────────────────────────

    async def ensure_graph(self, tenant_id: str, corpus_id: str) -> None:
        """Create the per-corpus AGE graph if it does not already exist."""
        graph = self._graph_name(tenant_id, corpus_id)
        async with self._conn() as conn:
            try:
                await conn.execute(f"SELECT create_graph('{graph}')")
                logger.info("Created AGE graph '%s'", graph)
            except Exception as exc:
                if "already exists" in str(exc).lower():
                    logger.debug("AGE graph '%s' already exists", graph)
                else:
                    raise

    async def delete_corpus_graph(self, tenant_id: str, corpus_id: str) -> None:
        """Drop the entire AGE graph for a corpus."""
        graph = self._graph_name(tenant_id, corpus_id)
        async with self._conn() as conn:
            await conn.execute(f"SELECT drop_graph('{graph}', true)")
        logger.info("Dropped AGE graph '%s'", graph)

    async def delete_document_vertices(
        self, tenant_id: str, corpus_id: str, document_id: str
    ) -> None:
        """Remove all vertices (and their edges) for one document."""
        graph = self._graph_name(tenant_id, corpus_id)
        doc_esc = document_id.replace('"', '\\"')
        cypher = f'MATCH (v {{document_id: "{doc_esc}"}}) DETACH DELETE v'
        async with self._conn() as conn:
            await conn.execute(self._cypher_sql(graph, cypher, "r agtype"))

    # ── Write ─────────────────────────────────────────────────────────────────

    async def _upsert_vertex(
        self,
        graph: str,
        nx_id: str,
        label: str,
        name: str,
        props: dict[str, Any],
    ) -> str:
        """MERGE vertex by (nx_id, corpus_id); return AGE uuid property."""
        vertex_uuid = str(_uuid.uuid4())
        name_esc   = name.replace("\\", "\\\\").replace('"', '\\"')
        nx_id_esc  = str(nx_id).replace('"', '\\"')
        corpus_esc = str(props.get("corpus_id", "")).replace('"', '\\"')
        safe_label = _sanitize_label(label)

        cypher = (
            f'MERGE (v:{safe_label} {{nx_id: "{nx_id_esc}", corpus_id: "{corpus_esc}"}}) '
            f'SET v.uuid = COALESCE(v.uuid, "{vertex_uuid}"), '
            f'v.name = "{name_esc}", '
            f'v.label = "{safe_label}" '
            f'RETURN v.uuid'
        )
        async with self._conn() as conn:
            rows = await conn.fetch(self._cypher_sql(graph, cypher, "uuid agtype"))
        return _unquote_agtype(rows[0]["uuid"]) if rows else vertex_uuid

    async def _add_edge(
        self,
        graph: str,
        src_uuid: str,
        tgt_uuid: str,
        rel_type: str,
        props: dict[str, Any],
    ) -> None:
        """Create a directed edge between two vertices matched by uuid property."""
        safe_rel   = _sanitize_rel_type(rel_type)
        src_esc    = src_uuid.replace('"', '\\"')
        tgt_esc    = tgt_uuid.replace('"', '\\"')
        corpus_esc = str(props.get("corpus_id", "")).replace('"', '\\"')
        doc_esc    = str(props.get("document_id", "")).replace('"', '\\"')

        cypher = (
            f'MATCH (s {{uuid: "{src_esc}"}}), (t {{uuid: "{tgt_esc}"}}) '
            f'CREATE (s)-[r:{safe_rel} {{'
            f'corpus_id: "{corpus_esc}", document_id: "{doc_esc}"'
            f'}}]->(t)'
        )
        async with self._conn() as conn:
            await conn.execute(self._cypher_sql(graph, cypher, "r agtype"))

    async def import_docling_graph(
        self,
        context: Any,   # docling_graph.pipeline.context.PipelineContext
        corpus_id: str,
        tenant_id: str,
        document_id: str,
    ) -> tuple[int, int]:
        """Import a docling-graph PipelineContext into Apache AGE.

        Iterates context.knowledge_graph (NetworkX DiGraph) directly.
        Do NOT use CypherExporter — it generates Neo4j syntax incompatible
        with AGE's ag_catalog.cypher() SQL wrapper.

        Returns (node_count, edge_count).
        """
        graph_name = self._graph_name(tenant_id, corpus_id)
        nx_graph: DiGraph = context.knowledge_graph

        node_id_map: dict[str, str] = {}

        for nx_id, attrs in nx_graph.nodes(data=True):
            label = attrs.get("label", "Entity")
            name  = str(attrs.get("name") or attrs.get("id") or nx_id)
            props = {
                k: str(v) for k, v in attrs.items()
                if k != "label" and v is not None
            }
            props["corpus_id"]   = corpus_id
            props["document_id"] = document_id
            props["tenant_id"]   = tenant_id

            age_uuid = await self._upsert_vertex(graph_name, str(nx_id), label, name, props)
            node_id_map[str(nx_id)] = age_uuid

        edge_count = 0
        for src_nx, tgt_nx, edge_attrs in nx_graph.edges(data=True):
            rel_type = edge_attrs.get("label", "RELATED_TO")
            src_uuid = node_id_map.get(str(src_nx))
            tgt_uuid = node_id_map.get(str(tgt_nx))
            if src_uuid and tgt_uuid:
                await self._add_edge(
                    graph_name, src_uuid, tgt_uuid, rel_type,
                    {"corpus_id": corpus_id, "document_id": document_id},
                )
                edge_count += 1

        logger.info(
            "Imported %d nodes, %d edges into AGE graph '%s'",
            len(nx_graph.nodes), edge_count, graph_name,
        )
        return len(nx_graph.nodes), edge_count

    # ── Read ──────────────────────────────────────────────────────────────────

    async def run_cypher_query(
        self, cypher: str, corpus_id: str, tenant_id: str
    ) -> str:
        """Execute a read-only MATCH query. Returns pipe-delimited table string."""
        if re.search(
            r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH|FOREACH|LOAD\s+CSV)\b",
            cypher, re.IGNORECASE,
        ):
            return "Error: only read-only MATCH queries are permitted."

        graph_name = self._graph_name(tenant_id, corpus_id)
        aliases    = _parse_return_aliases(cypher)
        as_clause  = ", ".join(f"c{i} agtype" for i in range(len(aliases)))

        async with self._conn() as conn:
            try:
                rows = await conn.fetch(self._cypher_sql(graph_name, cypher, as_clause))
            except Exception as exc:
                return f"Cypher error: {exc}"

        if not rows:
            return "No results."

        header = " | ".join(aliases)
        sep    = "-" * max(len(header), 10)
        lines  = [header, sep]
        for row in rows:
            vals = [_unquote_agtype(row[f"c{i}"]) for i in range(len(aliases))]
            lines.append(" | ".join(vals))
        n = len(rows)
        lines.append(f"\n({n} row{'s' if n != 1 else ''})")
        return "\n".join(lines)

    async def search_as_context(
        self, query: str, corpus_id: str, tenant_id: str, limit: int = 15
    ) -> str:
        """Search entities + relationships; return LLM-ready context string."""
        graph_name = self._graph_name(tenant_id, corpus_id)
        query_esc  = query.lower().replace("\\", "\\\\").replace('"', '\\"')
        cypher = (
            f'MATCH (e)-[r]->(t) '
            f'WHERE toLower(e.name) CONTAINS "{query_esc}" '
            f'   OR toLower(t.name) CONTAINS "{query_esc}" '
            f'RETURN e.name, e.label, type(r) AS rel, t.name, t.label '
            f'LIMIT {limit}'
        )
        as_clause = "src_name agtype, src_label agtype, rel agtype, tgt_name agtype, tgt_label agtype"

        async with self._conn() as conn:
            try:
                rows = await conn.fetch(self._cypher_sql(graph_name, cypher, as_clause))
            except Exception:
                return "No graph context available."

        if not rows:
            return "No relevant entities found in knowledge graph."

        lines = [
            f"- [{_unquote_agtype(r['src_label'])}] {_unquote_agtype(r['src_name'])} "
            f"--{_unquote_agtype(r['rel'])}--> "
            f"[{_unquote_agtype(r['tgt_label'])}] {_unquote_agtype(r['tgt_name'])}"
            for r in rows
        ]
        return "## Knowledge Graph — Facts\n" + "\n".join(lines)

    async def get_graph_stats(self, tenant_id: str, corpus_id: str) -> dict[str, Any]:
        """Return vertex and edge counts by type."""
        graph_name = self._graph_name(tenant_id, corpus_id)
        async with self._conn() as conn:
            v_rows = await conn.fetch(
                self._cypher_sql(
                    graph_name,
                    "MATCH (e) RETURN e.label, count(*) AS cnt",
                    "label agtype, cnt agtype",
                )
            )
            e_rows = await conn.fetch(
                self._cypher_sql(
                    graph_name,
                    "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS cnt",
                    "rel_type agtype, cnt agtype",
                )
            )
        return {
            "total_entities":      sum(int(_unquote_agtype(r["cnt"])) for r in v_rows),
            "total_relationships": sum(int(_unquote_agtype(r["cnt"])) for r in e_rows),
            "entities_by_type":    {_unquote_agtype(r["label"]): int(_unquote_agtype(r["cnt"])) for r in v_rows},
            "relationships_by_type": {_unquote_agtype(r["rel_type"]): int(_unquote_agtype(r["cnt"])) for r in e_rows},
        }
